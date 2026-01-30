from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import Settings
from opsbrain_graph.agents import AgentResult, AgentTask, AgentRecommendation, AgentMetadata
from opsbrain_graph.state import GraphState, DiagnosisSummary, PendingActionProposal
from utils.llm import get_llm

if TYPE_CHECKING:
    from opsbrain_graph.agents import BaseAgent

logger = logging.getLogger("supervisor")


# =============================================================================
# Dynamic Planning Prompt Generator
# =============================================================================


def generate_planning_prompt(agent_metadata: dict[str, AgentMetadata]) -> str:
    """
    Generate the planning system prompt dynamically from agent metadata.

    This allows the prompt to stay in sync with actual agent capabilities.
    """
    lines = [
        "You are an AI Operations Supervisor for an e-commerce business.",
        "Your job is to analyze the user's question and create a task plan by assigning work to specialist agents.",
        "",
        "## Available Agents and Their Capabilities:",
        "",
    ]

    # Add each agent's metadata
    for idx, (name, meta) in enumerate(sorted(agent_metadata.items()), 1):
        lines.append(f"### {idx}. {meta.display_name} Agent")
        lines.append(f"- {meta.description}")

        if meta.capabilities:
            lines.append("- Capabilities (use the name as 'mode' parameter):")
            for cap in meta.capabilities:
                params_desc = ""
                if cap.parameters:
                    params_list = [f"{k}: {v}" for k, v in cap.parameters.items()]
                    params_desc = f" (parameters: {'; '.join(params_list)})"
                lines.append(f'  * mode="{cap.name}": {cap.description}{params_desc}')
                if cap.example_queries:
                    lines.append(f"    Examples: {'; '.join(cap.example_queries[:2])}")

        if meta.keywords:
            lines.append(f"- Trigger keywords: {', '.join(meta.keywords[:7])}")

        lines.append("")

    # Add task assignment rules with comprehensive examples
    lines.extend(
        [
            "## Task Assignment Rules:",
            "1. Assign the MINIMUM number of agents needed to answer the question",
            "2. Multiple agents can work in PARALLEL if their tasks are independent",
            "3. For 'why' questions, include the HISTORIAN agent",
            "4. For product-related issues, consider both SALES and INVENTORY",
            "5. Be specific with parameters - extract numbers, time windows, and filters from the query",
            "6. Use agent keywords to help identify which agents are relevant",
            "",
            "## SLIMMED AGENT ARCHITECTURE:",
            "Each specialized agent has LIMITED capabilities (2 core tools each).",
            "For complex queries that don't fit core capabilities, the agent will return 'cannot_handle'",
            "and the system will automatically route to DATA_ANALYST with HITL approval.",
            "",
            "## Agent Core Capabilities:",
            "- SALES: summary, top_products (for period comparison, regional, channel → routes to analyst)",
            "- INVENTORY: check_stock, low_stock_scan (for predictions, top-sellers stock → routes to analyst)",
            "- MARKETING: campaign_spend, calculate_roas (for underperforming, comparison → routes to analyst)",
            "- SUPPORT: sentiment_analysis, ticket_trends (for common issues, complaint comparison → routes to analyst)",
            "- DATA_ANALYST: execute_sql - handles complex queries with HITL approval",
            "- HISTORIAN: query - retrieves past incidents from memory",
            "",
            "## CRITICAL RULES FOR MODE SELECTION:",
            "7. ALWAYS specify the 'mode' parameter - match it to the capability name",
            "8. For simple 'sales summary', 'how did sales do' → mode='summary'",
            "9. For 'top products', 'best sellers' → mode='top_products'",
            "10. For 'low stock', 'out of stock', 'close to stockout' → mode='low_stock_scan'",
            "11. For 'check stock for product X' → mode='check_stock' with product_ids",
            "12. For 'campaign spend', 'ad spend' → mode='campaign_spend'",
            "13. For 'ROAS', 'return on ad spend' → mode='calculate_roas'",
            "14. For 'sentiment', 'how is support' → mode='sentiment_analysis'",
            "15. For 'ticket trends', 'support trends' → mode='ticket_trends'",
            "",
            "## Complex Queries (Agent will auto-route to DATA_ANALYST):",
            "- 'Compare yesterday to last week' → Try sales first, will route to analyst",
            "- 'Underperforming campaigns' → Try marketing first, will route to analyst",
            "- 'Common customer issues' → Try support first, will route to analyst",
            "- 'Predict stockouts' → Try inventory first, will route to analyst",
            "",
            "## Cross-Domain Questions:",
            "For questions that span multiple domains, assign multiple agents:",
            "- 'Was the sales drop caused by inventory, marketing, or support issues?' → sales, inventory, marketing, support agents",
            "- 'Show all contributing factors' → sales, inventory, marketing, support agents",
            "- 'Correlate complaints with sales' → sales + support agents",
            "",
            "## Output Format:",
            "Return a JSON array of tasks. Each task must have:",
            "- agent: One of " + ", ".join(f'"{name}"' for name in sorted(agent_metadata.keys())),
            "- objective: Clear description of what the agent should accomplish",
            "- parameters: Dict with 'mode' (REQUIRED) and 'query' (pass original question), plus window_days, limit, product_ids, etc.",
            "- priority: 1 (highest) to 5 (lowest) - for execution ordering",
            "",
            "## Examples:",
            "",
            "Question: 'What are the top selling products?'",
            '[{"agent": "sales", "objective": "Find top selling products", "parameters": {"mode": "top_products", "query": "What are the top selling products?", "window_days": 7, "limit": 5}, "priority": 1}]',
            "",
            "Question: 'Compare yesterday sales to last week'",
            '[{"agent": "sales", "objective": "Compare sales periods", "parameters": {"mode": "summary", "query": "Compare yesterday sales to last week"}, "priority": 1}]',
            "",
            "Question: 'Which products are close to stock-out?'",
            '[{"agent": "inventory", "objective": "Scan for low stock products", "parameters": {"mode": "low_stock_scan", "query": "Which products are close to stock-out?"}, "priority": 1}]',
            "",
            "Question: 'What is our campaign ROAS?'",
            '[{"agent": "marketing", "objective": "Calculate ROAS", "parameters": {"mode": "calculate_roas", "query": "What is our campaign ROAS?", "window_days": 7}, "priority": 1}]',
            "",
            "Question: 'What is customer sentiment?'",
            '[{"agent": "support", "objective": "Analyze sentiment", "parameters": {"mode": "sentiment_analysis", "query": "What is customer sentiment?", "window_days": 7}, "priority": 1}]',
            "",
            "Question: 'Summarize yesterday business health'",
            "[",
            '  {"agent": "sales", "objective": "Get sales summary", "parameters": {"mode": "summary", "query": "Summarize yesterday business health", "window_days": 1}, "priority": 1},',
            '  {"agent": "inventory", "objective": "Check for stock issues", "parameters": {"mode": "low_stock_scan", "query": "Summarize yesterday business health"}, "priority": 1},',
            '  {"agent": "marketing", "objective": "Get campaign spend", "parameters": {"mode": "campaign_spend", "query": "Summarize yesterday business health"}, "priority": 1},',
            '  {"agent": "support", "objective": "Analyze support sentiment", "parameters": {"mode": "sentiment_analysis", "query": "Summarize yesterday business health", "window_days": 1}, "priority": 1}',
            "]",
            "",
            "IMPORTANT: Return ONLY the JSON array, no additional text or markdown.",
            "IMPORTANT: Always include 'query' parameter with the original user question for agent fallback routing.",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Fallback static prompt (used if no agents registered)
# =============================================================================

PLANNING_SYSTEM_PROMPT = """You are an AI Operations Supervisor for an e-commerce business.
Your job is to analyze the user's question and create a task plan by assigning work to specialist agents.

## Available Agents and Their Capabilities:

### 1. SALES Agent
- Analyzes revenue trends, sales performance, and detects anomalies
- Can identify top-selling products
- Capabilities:
  * mode: "trends" - Analyze revenue trends over time (parameters: window_days, group_by)
  * mode: "top_products" - Find best-selling products (parameters: window_days, limit)

### 2. INVENTORY Agent
- Monitors stock levels and identifies low-stock items
- Capabilities:
  * Check stock levels for specific products (parameters: product_ids)
  * Detect items below low_stock_threshold

### 3. MARKETING Agent
- Evaluates campaign performance, ad spend, ROI
- Capabilities:
  * Analyze campaign spend efficiency (parameters: window_days)
  * Track clicks, conversions, ROAS

### 4. SUPPORT Agent
- Analyzes customer support tickets and sentiment
- Capabilities:
  * Summarize support sentiment (parameters: window_days, product_id)
  * Detect issue spikes and negative sentiment trends

### 5. HISTORIAN Agent
- Retrieves similar past incidents from memory for context
- Use when user asks "why", wants explanations, or needs historical context
- Capabilities:
  * mode: "query" - Search past incidents (parameters: query)

### 6. DATA_ANALYST Agent
- Performs custom SQL queries for complex analysis
- Use for questions that don't fit other agents
- Capabilities:
  * Execute analytical queries (parameters: query_type, filters)

## Task Assignment Rules:
1. Assign the MINIMUM number of agents needed to answer the question
2. Multiple agents can work in PARALLEL if their tasks are independent
3. For "why" questions, include the HISTORIAN agent
4. For product-related issues, consider both SALES and INVENTORY
5. Be specific with parameters - extract numbers, time windows, and filters from the query

## Output Format:
Return a JSON array of tasks. Each task must have:
- agent: One of "sales", "inventory", "marketing", "support", "historian", "data_analyst"
- objective: Clear description of what the agent should accomplish
- parameters: Dict with mode, window_days, limit, product_ids, etc. as needed
- priority: 1 (highest) to 5 (lowest) - for execution ordering

Example:
[
  {"agent": "sales", "objective": "Find top 5 selling products", "parameters": {"mode": "top_products", "window_days": 7, "limit": 5}, "priority": 1},
  {"agent": "inventory", "objective": "Check stock for top sellers", "parameters": {"product_ids": []}, "priority": 2}
]

IMPORTANT: Return ONLY the JSON array, no additional text or markdown."""


# =============================================================================
# Pydantic models for structured plan output
# =============================================================================


class PlannedTask(BaseModel):
    """A single task in the battle plan."""

    agent: str = Field(
        ..., description="Agent name: sales, inventory, marketing, support, historian, data_analyst"
    )
    objective: str = Field(..., description="What the agent should accomplish")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific parameters"
    )
    priority: int = Field(default=1, ge=1, le=5, description="Execution priority (1=highest)")


class TaskPlan(BaseModel):
    """The complete task plan from the LLM."""

    tasks: list[PlannedTask] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Brief explanation of the plan")


SYNTHESIS_SYSTEM_PROMPT = """You are an AI Operations Analyst for an e-commerce business. 
Your job is to analyze data from various agents and provide clear, actionable insights.

Based on the collected findings and insights from specialist agents, provide:
1. A clear, concise answer to the user's question
2. Key findings summarized in plain language
3. Recommended actions if any issues are detected

Be specific with numbers and percentages. If there's a problem, explain potential causes.
If you don't have enough data to answer, say so clearly.

Format your response in a clear, professional manner."""


@dataclass
class SupervisorOutput:
    summary: DiagnosisSummary
    answer: str
    diagnostics: list[str]
    pending_actions: list[PendingActionProposal]


class Supervisor:
    """
    Responsible for high-level planning (battle plan) and synthesis of agent outputs.
    """

    def __init__(
        self,
        settings: Settings,
        agent_metadata: dict[str, AgentMetadata] | None = None,
    ) -> None:
        self.settings = settings
        self._llm = None
        self._agent_metadata = agent_metadata or {}
        self._planning_prompt: str | None = None

    def register_agents(self, agents: dict[str, "BaseAgent"]) -> None:
        """Register agents and build planning prompt from their metadata."""
        self._agent_metadata = {name: agent.get_metadata() for name, agent in agents.items()}
        self._planning_prompt = None  # Force regeneration

    @property
    def planning_prompt(self) -> str:
        """Get the planning prompt, generating from metadata if available."""
        if self._planning_prompt is None:
            if self._agent_metadata:
                self._planning_prompt = generate_planning_prompt(self._agent_metadata)
                logger.info(
                    "Generated dynamic planning prompt from %d agents", len(self._agent_metadata)
                )
            else:
                self._planning_prompt = PLANNING_SYSTEM_PROMPT
                logger.info("Using static fallback planning prompt")
        return self._planning_prompt

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def initialize_state(
        self,
        user_query: str,
        conversation_history: Sequence[dict[str, Any]] | None = None,
    ) -> GraphState:
        return GraphState(
            user_query=user_query,
            conversation_history=list(conversation_history or []),
            battle_plan=[],
            agent_findings={},
            agent_insights={},
            recommendations=[],
            memory_context=[],
            diagnosis=None,
            pending_action_proposals=[],
            hitl_wait=False,
            system_warnings=[],
            metadata={},
            # Re-planning control
            replan_count=0,
            max_replans=2,
            needs_replan=False,
            replan_reason=None,
        )

    async def plan(self, state: GraphState) -> list[AgentTask]:
        """
        Use LLM to analyze the user query and create a task plan.

        Returns a list of AgentTask objects to be executed.
        """
        query = state["user_query"]

        try:
            tasks = await self._llm_plan(query, state)
            if tasks:
                state["battle_plan"] = tasks
                return tasks
        except Exception as exc:
            logger.warning("LLM planning failed, falling back to keyword matching: %s", exc)

        # Fallback to keyword-based planning if LLM fails
        tasks = self._keyword_plan(state)
        state["battle_plan"] = tasks
        return tasks

    async def _llm_plan(self, query: str, state: GraphState) -> list[AgentTask]:
        """Use LLM to generate the task plan."""

        # Build context with any additional info
        context_parts = [f"User Question: {query}"]

        if state.get("conversation_history"):
            context_parts.append("\nRecent conversation context:")
            for msg in state["conversation_history"][-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                context_parts.append(f"  {role}: {content}")

        if state.get("metadata"):
            context_parts.append(f"\nMetadata: {json.dumps(state['metadata'])}")

        user_content = "\n".join(context_parts)

        messages = [
            SystemMessage(content=self.planning_prompt),  # Use dynamic prompt
            HumanMessage(content=user_content),
        ]

        response = await self.llm.ainvoke(messages)
        raw_plan = response.content.strip()

        # Parse the JSON response
        tasks = self._parse_plan_response(raw_plan, state)

        if tasks:
            logger.info(
                "LLM planned %d tasks: %s",
                len(tasks),
                [(t.agent, t.parameters) for t in tasks],
            )

        return tasks

    def _parse_plan_response(self, raw_response: str, state: GraphState) -> list[AgentTask]:
        """Parse the LLM's JSON response into AgentTask objects."""

        # Clean up response - remove markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Remove markdown code block
            lines = cleaned.split("\n")
            # Find the start and end of code block
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM plan as JSON: %s", exc)
            return []

        # Handle both array and object formats
        if isinstance(parsed, dict):
            if "tasks" in parsed:
                parsed = parsed["tasks"]
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            logger.warning("LLM plan is not a list: %s", type(parsed))
            return []

        # Valid agent names
        valid_agents = {"sales", "inventory", "marketing", "support", "historian", "data_analyst"}

        tasks: list[AgentTask] = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            agent = item.get("agent", "").lower().replace(" ", "_")
            if agent not in valid_agents:
                logger.warning("Invalid agent in plan: %s", agent)
                continue

            objective = item.get("objective", f"Execute {agent} task")
            parameters = item.get("parameters", {})
            priority = item.get("priority", idx + 1)

            # Determine result slot based on agent
            if agent == "historian":
                result_slot = "memory_context"
            else:
                result_slot = f"agent_findings.{agent}"

            task = AgentTask(
                agent=agent,
                objective=objective,
                parameters=parameters,
                result_slot=result_slot,
            )
            tasks.append((priority, task))

        # Sort by priority and extract tasks
        tasks.sort(key=lambda x: x[0])
        return [t for _, t in tasks]

    def _keyword_plan(self, state: GraphState) -> list[AgentTask]:
        """Fallback keyword-based planning when LLM is unavailable."""
        query = state["user_query"].lower()
        tasks: list[AgentTask] = []

        # Top products query (must check before general sales)
        if any(kw in query for kw in ("top", "best", "highest", "most sold", "best selling")):
            if any(kw in query for kw in ("product", "item", "sku", "selling")):
                limit = 5  # default
                import re

                match = re.search(r"top\s*(\d+)", query)
                if match:
                    limit = int(match.group(1))
                tasks.append(
                    AgentTask(
                        agent="sales",
                        objective="Find top selling products.",
                        parameters={"mode": "top_products", "window_days": 7, "limit": limit},
                        result_slot="agent_findings.sales",
                    )
                )

        # Sales-related queries (general trends)
        if not tasks and any(
            kw in query for kw in ("sale", "revenue", "drop", "trend", "income", "earning", "money")
        ):
            tasks.append(
                AgentTask(
                    agent="sales",
                    objective="Analyze revenue trends and detect anomalies.",
                    parameters={"mode": "trends", "window_days": 7, "group_by": "day"},
                    result_slot="agent_findings.sales",
                )
            )

        # Inventory-related queries
        if any(kw in query for kw in ("stock", "inventory", "out of stock", "restock", "supply")):
            tasks.append(
                AgentTask(
                    agent="inventory",
                    objective="Check stock levels for key products.",
                    parameters={
                        "product_ids": state.get("metadata", {}).get("focus_product_ids", [1, 2, 3])
                    },
                    result_slot="agent_findings.inventory",
                )
            )

        # Marketing-related queries
        if any(kw in query for kw in ("campaign", "ad", "marketing", "roas", "spend", "promotion")):
            tasks.append(
                AgentTask(
                    agent="marketing",
                    objective="Evaluate campaign spend efficiency.",
                    parameters={"window_days": 7},
                    result_slot="agent_findings.marketing",
                )
            )

        # Support-related queries
        if any(
            kw in query
            for kw in ("ticket", "support", "sentiment", "complaint", "customer", "issue")
        ):
            tasks.append(
                AgentTask(
                    agent="support",
                    objective="Summarize support sentiment and issue spikes.",
                    parameters={"window_days": 7},
                    result_slot="agent_findings.support",
                )
            )

        # Always include historian for context on "why" questions
        if any(kw in query for kw in ("why", "reason", "cause", "explain", "happened")):
            tasks.append(
                AgentTask(
                    agent="historian",
                    objective="Retrieve similar past incidents for context.",
                    parameters={"mode": "query", "query": state["user_query"]},
                    result_slot="memory_context",
                )
            )

        # Default: if no specific agent matched, use sales as general check
        if not tasks:
            tasks.append(
                AgentTask(
                    agent="sales",
                    objective="General sales health check.",
                    parameters={"window_days": 7, "group_by": "day"},
                    result_slot="agent_findings.sales",
                )
            )

        return tasks

    def incorporate_agent_result(
        self,
        state: GraphState,
        agent_name: str,
        result: AgentResult,
    ) -> None:
        if result.status == "success":
            # Store findings
            if "agent_findings" not in state:
                state["agent_findings"] = {}
            state["agent_findings"][agent_name] = result.findings

            # Store insights
            if "agent_insights" not in state:
                state["agent_insights"] = {}
            state["agent_insights"][agent_name] = result.insights

            # Collect recommendations
            if "recommendations" not in state:
                state["recommendations"] = []
            state["recommendations"].extend(result.recommendations)
        elif result.status == "cannot_handle":
            # Agent indicates it cannot handle this query - needs DataAnalyst routing
            logger.info(
                "Agent %s returned cannot_handle, will route to data_analyst",
                agent_name,
            )
            if "cannot_handle_agents" not in state:
                state["cannot_handle_agents"] = []
            state["cannot_handle_agents"].append(
                {
                    "agent": agent_name,
                    "query": result.findings.get("query", state.get("user_query", "")),
                    "reason": result.findings.get("reason", "Requires complex analysis"),
                }
            )
            # Also store insights for transparency
            if "agent_insights" not in state:
                state["agent_insights"] = {}
            state["agent_insights"][agent_name] = result.insights
        else:
            if "system_warnings" not in state:
                state["system_warnings"] = []
            state["system_warnings"].append(f"{agent_name} agent failed: {result.errors}")

    def evaluate_results(self, state: GraphState) -> bool:
        """
        Evaluate agent results to determine if re-planning is needed.

        Returns True if results are sufficient, False if re-planning needed.
        """
        replan_count = state.get("replan_count", 0)
        max_replans = state.get("max_replans", 2)

        # Don't exceed max replans
        if replan_count >= max_replans:
            logger.info("Max replans (%d) reached, proceeding to synthesis", max_replans)
            state["needs_replan"] = False
            return True

        battle_plan = state.get("battle_plan", [])
        agent_findings = state.get("agent_findings", {})
        system_warnings = state.get("system_warnings", [])
        cannot_handle_agents = state.get("cannot_handle_agents", [])

        # Check for agents that returned cannot_handle - route to data_analyst
        if cannot_handle_agents:
            # Check if data_analyst already ran
            if "data_analyst" in agent_findings:
                # Data analyst already provided results, don't replan
                logger.info("Data analyst already ran, proceeding to synthesis")
                state["needs_replan"] = False
                return True

            # Need to route to data_analyst for complex queries
            state["needs_replan"] = True
            agent_names = [c["agent"] for c in cannot_handle_agents]
            state["replan_reason"] = (
                f"Agents {agent_names} cannot handle query, routing to data_analyst"
            )
            state["route_to_analyst"] = True
            logger.info("Re-planning needed: routing to data_analyst for complex query")
            return False

        # Check for failed agents that were critical
        failed_agents = set()
        for warning in system_warnings:
            for task in battle_plan:
                if task.agent in warning and "failed" in warning.lower():
                    failed_agents.add(task.agent)

        # Check if we got results from at least one agent
        if not agent_findings:
            state["needs_replan"] = True
            state["replan_reason"] = "No agents returned findings"
            logger.warning("Re-planning needed: no agent findings")
            return False

        # Check if critical agents failed (e.g., the first/highest priority task)
        if battle_plan and battle_plan[0].agent in failed_agents:
            state["needs_replan"] = True
            state["replan_reason"] = f"Primary agent '{battle_plan[0].agent}' failed"
            logger.warning("Re-planning needed: primary agent failed")
            return False

        # Check for empty findings (agent returned but with no useful data)
        empty_findings = []
        for agent_name, findings in agent_findings.items():
            if self._is_empty_result(findings):
                empty_findings.append(agent_name)

        # If all agents returned empty, try re-planning
        if empty_findings and len(empty_findings) == len(agent_findings):
            state["needs_replan"] = True
            state["replan_reason"] = f"All agents returned empty results: {empty_findings}"
            logger.warning("Re-planning needed: all agents returned empty results")
            return False

        # Results are sufficient
        state["needs_replan"] = False
        state["replan_reason"] = None
        return True

    def _is_empty_result(self, findings: dict[str, Any]) -> bool:
        """Check if findings are effectively empty."""
        if not findings:
            return True

        # Check common patterns for empty results
        for key, value in findings.items():
            if isinstance(value, list) and len(value) > 0:
                return False
            if isinstance(value, dict) and len(value) > 0:
                return False
            if isinstance(value, (int, float)) and value != 0:
                return False
            if isinstance(value, str) and value.strip():
                return False

        return True

    async def replan(self, state: GraphState) -> list[AgentTask]:
        """
        Create a new plan based on what failed or returned empty.

        This considers the previous failures and tries alternative approaches.
        Handles cannot_handle routing to data_analyst specially.
        """
        replan_count = state.get("replan_count", 0)
        state["replan_count"] = replan_count + 1

        replan_reason = state.get("replan_reason", "Unknown reason")

        # Check if we need to route to data_analyst due to cannot_handle
        if state.get("route_to_analyst"):
            logger.info("Routing to data_analyst for complex query")
            cannot_handle_agents = state.get("cannot_handle_agents", [])
            original_query = state.get("user_query", "")

            # Build query context from cannot_handle agents
            query_context = original_query
            if cannot_handle_agents:
                reasons = [c.get("reason", "") for c in cannot_handle_agents]
                query_context = (
                    f"{original_query} (Note: specialized agents indicated: {'; '.join(reasons)})"
                )

            # Create task for data_analyst with the original query
            task = AgentTask(
                agent="data_analyst",
                objective=f"Generate custom SQL to answer: {original_query}",
                parameters={
                    "mode": "analyze",
                    "query": original_query,
                },
                result_slot="agent_findings.data_analyst",
            )
            state["battle_plan"] = [task]
            state["route_to_analyst"] = False  # Clear the flag
            return [task]

        failed_agents = set()

        # Identify failed agents from warnings
        for warning in state.get("system_warnings", []):
            for agent_name in self._agent_metadata.keys():
                if agent_name in warning.lower():
                    failed_agents.add(agent_name)

        # Build context for LLM re-planning
        context_parts = [
            f"User Question: {state['user_query']}",
            f"\nRe-planning attempt #{state['replan_count']} due to: {replan_reason}",
        ]

        if failed_agents:
            context_parts.append(
                f"Failed agents to avoid or retry differently: {list(failed_agents)}"
            )

        previous_findings = state.get("agent_findings", {})
        if previous_findings:
            context_parts.append(
                f"\nPartial results already collected from: {list(previous_findings.keys())}"
            )
            context_parts.append("Consider if additional agents could help complete the answer.")

        user_content = "\n".join(context_parts)

        try:
            messages = [
                SystemMessage(content=self.planning_prompt),
                HumanMessage(content=user_content),
            ]

            response = await self.llm.ainvoke(messages)
            raw_plan = response.content.strip()

            tasks = self._parse_plan_response(raw_plan, state)

            # Filter out agents we already have good results from
            tasks = [
                t for t in tasks if t.agent not in previous_findings or t.agent in failed_agents
            ]

            if tasks:
                logger.info("Re-planned %d new tasks: %s", len(tasks), [t.agent for t in tasks])
                state["battle_plan"] = tasks
                return tasks

        except Exception as exc:
            logger.warning("LLM re-planning failed: %s", exc)

        # Fallback: retry failed agents with different approach or use data_analyst
        fallback_tasks = []
        if failed_agents and "data_analyst" not in failed_agents:
            fallback_tasks.append(
                AgentTask(
                    agent="data_analyst",
                    objective=f"Analyze data to answer: {state['user_query']}",
                    parameters={"statement": "SELECT 1", "params": None},  # Placeholder
                    result_slot="agent_findings.data_analyst",
                )
            )

        if fallback_tasks:
            state["battle_plan"] = fallback_tasks
            return fallback_tasks

        # No fallback possible, proceed with what we have
        state["needs_replan"] = False
        return []

    async def synthesize(self, state: GraphState) -> SupervisorOutput:
        """Use LLM to synthesize agent findings into a coherent answer."""

        user_query = state["user_query"]
        agent_findings = state.get("agent_findings", {})
        agent_insights = state.get("agent_insights", {})
        recommendations = state.get("recommendations", [])
        memory_context = state.get("memory_context", [])
        warnings = state.get("system_warnings", [])

        # Build context for LLM
        context_parts = []

        context_parts.append(f"USER QUESTION: {user_query}\n")

        if agent_findings:
            context_parts.append("COLLECTED DATA FROM AGENTS:")
            for agent_name, findings in agent_findings.items():
                context_parts.append(f"\n--- {agent_name.upper()} AGENT FINDINGS ---")
                context_parts.append(json.dumps(findings, indent=2, default=str))

        if agent_insights:
            context_parts.append("\n\nAGENT INSIGHTS:")
            for agent_name, insights in agent_insights.items():
                context_parts.append(f"\n{agent_name.upper()}:")
                for insight in insights:
                    context_parts.append(f"  • {insight}")

        if memory_context:
            context_parts.append("\n\nHISTORICAL CONTEXT (Similar Past Incidents):")
            for memory in memory_context:
                context_parts.append(f"  • {memory}")

        if warnings:
            context_parts.append("\n\nWARNINGS:")
            for warning in warnings:
                context_parts.append(f"  ⚠️ {warning}")

        context = "\n".join(context_parts)

        # Use LLM to generate answer
        try:
            messages = [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]

            response = await self.llm.ainvoke(messages)
            answer = response.content

        except Exception as exc:
            logger.exception("LLM synthesis failed: %s", exc)
            # Fallback to basic synthesis
            answer = self._fallback_synthesis(agent_insights, warnings)

        # Build diagnosis summary
        all_insights = []
        for agent_name, insights in agent_insights.items():
            for insight in insights:
                all_insights.append(f"{agent_name}: {insight}")

        summary = DiagnosisSummary(
            narrative=answer,
            key_findings=all_insights,
            confidence=min(0.95, 0.5 + 0.1 * len(all_insights)),
        )
        state["diagnosis"] = summary

        # Convert recommendations to pending action proposals
        pending_actions = self._collect_pending_actions(state, recommendations)

        # Compile diagnostics
        diagnostics = [
            f"Agents executed: {', '.join(agent_findings.keys()) or 'none'}",
        ]
        if state.get("hitl_wait"):
            diagnostics.append("HITL pending actions detected.")
        if warnings:
            diagnostics.append(f"Warnings: {len(warnings)}")

        return SupervisorOutput(
            summary=summary,
            answer=answer,
            diagnostics=diagnostics,
            pending_actions=pending_actions,
        )

    def _fallback_synthesis(
        self,
        agent_insights: dict[str, list[str]],
        warnings: list[str],
    ) -> str:
        """Fallback synthesis when LLM is unavailable."""
        lines = []

        if agent_insights:
            lines.append("Based on the analysis:\n")
            for agent_name, insights in agent_insights.items():
                for insight in insights:
                    lines.append(f"• {insight}")
        else:
            lines.append("Investigation completed; awaiting more signals.")

        if warnings:
            lines.append("\nWarnings encountered:")
            for warning in warnings:
                lines.append(f"⚠️ {warning}")

        return "\n".join(lines)

    def _collect_pending_actions(
        self,
        state: GraphState,
        recommendations: list[AgentRecommendation],
    ) -> list[PendingActionProposal]:
        proposals: list[PendingActionProposal] = []

        for rec in recommendations:
            if rec.requires_approval:
                proposal = PendingActionProposal(
                    agent_name=(
                        rec.action_type.split("_")[0] if "_" in rec.action_type else "system"
                    ),
                    action_type=rec.action_type,
                    payload=rec.payload,
                    reasoning=rec.reasoning,
                    requires_approval=rec.requires_approval,
                )
                proposals.append(proposal)

        state["pending_action_proposals"] = proposals
        state["hitl_wait"] = bool(proposals)

        return proposals
