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
from prompts import get_prompt, render_prompt
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
    Uses templates from the centralized prompts folder.
    """
    # Load template sections from centralized prompts
    header = get_prompt("planning.system.header")
    rules = get_prompt("planning.system.rules")
    examples = get_prompt("planning.system.examples")

    # Build agent sections
    agent_lines = []
    for idx, (name, meta) in enumerate(sorted(agent_metadata.items()), 1):
        agent_lines.append(f"### {idx}. {meta.display_name} Agent")
        agent_lines.append(f"- {meta.description}")

        if meta.capabilities:
            agent_lines.append("- Capabilities (use the name as 'mode' parameter):")
            for cap in meta.capabilities:
                params_desc = ""
                if cap.parameters:
                    params_list = [f"{k}: {v}" for k, v in cap.parameters.items()]
                    params_desc = f" (parameters: {'; '.join(params_list)})"
                agent_lines.append(f'  * mode="{cap.name}": {cap.description}{params_desc}')
                if cap.example_queries:
                    agent_lines.append(f"    Examples: {'; '.join(cap.example_queries[:2])}")

        if meta.keywords:
            agent_lines.append(f"- Trigger keywords: {', '.join(meta.keywords[:7])}")

        agent_lines.append("")

    # Build agent names list for output format
    agent_names = ", ".join(f'"{name}"' for name in sorted(agent_metadata.keys()))
    output_format = render_prompt("planning.system.output_format", agent_names=agent_names)

    # Combine all sections
    full_prompt = header + "\n".join(agent_lines) + "\n" + rules + output_format + examples

    return full_prompt


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
        """Get the planning prompt, generating from metadata."""
        if self._planning_prompt is None:
            self._planning_prompt = generate_planning_prompt(self._agent_metadata)
            logger.info("Generated planning prompt from %d agents", len(self._agent_metadata))
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
        tasks = await self._llm_plan(query, state)
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

    def incorporate_agent_result(
        self,
        state: GraphState,
        agent_name: str,
        result: AgentResult,
    ) -> None:
        """Incorporate agent result into state."""
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
        """
        replan_count = state.get("replan_count", 0)
        state["replan_count"] = replan_count + 1

        replan_reason = state.get("replan_reason", "Unknown reason")
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

        # Fallback: use data_analyst for complex queries
        if "data_analyst" not in failed_agents and "data_analyst" not in previous_findings:
            fallback_task = AgentTask(
                agent="data_analyst",
                objective=f"Analyze data to answer: {state['user_query']}",
                parameters={"mode": "custom_analysis", "query": state["user_query"]},
                result_slot="agent_findings.data_analyst",
            )
            state["battle_plan"] = [fallback_task]
            return [fallback_task]

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
            synthesis_prompt = get_prompt("synthesis.system.template")
            messages = [
                SystemMessage(content=synthesis_prompt),
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

        # Calculate meaningful confidence score
        confidence = self._calculate_confidence(state, agent_findings, agent_insights, warnings)

        summary = DiagnosisSummary(
            narrative=answer,
            key_findings=all_insights,
            confidence=confidence,
        )
        state["diagnosis"] = summary

        # Convert recommendations to pending action proposals
        pending_actions = self._collect_pending_actions(state, recommendations)

        # Compile diagnostics
        diagnostics = [
            f"Agents executed: {', '.join(agent_findings.keys()) or 'none'}",
            f"Confidence score: {confidence:.2f}",
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

    def _calculate_confidence(
        self,
        state: GraphState,
        agent_findings: dict[str, Any],
        agent_insights: dict[str, list[str]],
        warnings: list[str],
    ) -> float:
        """
        Calculate a meaningful confidence score based on multiple factors.

        Factors (weights from reflection.yaml):
        - data_coverage (0.25): How many relevant agents provided data
        - data_quality (0.25): Whether agent responses contain substantive findings
        - cross_domain (0.20): Whether multiple domains were consulted
        - historical_context (0.15): Whether past incidents were checked
        - consistency (0.15): Whether findings are consistent (no conflicts)
        """
        score = 0.0

        # Factor 1: Data Coverage (0.25)
        # More agents responding = better coverage
        planned_agents = len(state.get("battle_plan", []))
        responding_agents = len(agent_findings)
        if planned_agents > 0:
            coverage_ratio = responding_agents / planned_agents
            score += 0.25 * coverage_ratio
        elif responding_agents > 0:
            score += 0.25  # Default full score if no plan

        # Factor 2: Data Quality (0.25)
        # Check if findings contain substantive data
        quality_score = 0.0
        for agent_name, findings in agent_findings.items():
            if not self._is_empty_result(findings):
                quality_score += 1.0
        if agent_findings:
            score += 0.25 * (quality_score / len(agent_findings))

        # Factor 3: Cross-Domain Analysis (0.20)
        # Multiple domain agents = richer analysis
        domain_agents = {"sales", "inventory", "marketing", "support"}
        consulted_domains = set(agent_findings.keys()) & domain_agents
        cross_domain_score = min(1.0, len(consulted_domains) / 2)  # 2+ domains = full score
        score += 0.20 * cross_domain_score

        # Factor 4: Historical Context (0.15)
        # Historian agent or memory context present
        has_history = "historian" in agent_findings or bool(state.get("memory_context"))
        score += 0.15 if has_history else 0.0

        # Factor 5: Consistency / Warnings (0.15)
        # Fewer warnings = higher confidence
        warning_count = len(warnings)
        consistency_score = max(0.0, 1.0 - (warning_count * 0.2))  # Each warning reduces by 0.2
        score += 0.15 * consistency_score

        # Clamp to valid range
        return max(0.1, min(0.95, score))

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

    async def reflect(self, state: GraphState) -> dict[str, Any]:
        """
        Reflect on the synthesized answer to detect missing information or weak conclusions.

        Returns a reflection result that may trigger re-synthesis or flag quality issues.
        """
        user_query = state["user_query"]
        agent_findings = state.get("agent_findings", {})
        diagnosis = state.get("diagnosis")
        answer = state.get("_final_answer", "")

        if not diagnosis or not answer:
            logger.warning("No synthesis output to reflect on")
            return {
                "is_complete": False,
                "missing_information": ["No analysis was generated"],
                "weak_conclusions": [],
                "confidence_adjustment": -0.2,
                "reasoning": "Synthesis did not produce an answer",
                "suggested_followup": None,
            }

        # Build context for reflection
        findings_summary = json.dumps(agent_findings, indent=2, default=str)

        reflection_prompt = render_prompt(
            "reflection.reflect.template",
            user_query=user_query,
            agent_findings=findings_summary,
            answer=answer,
            confidence=diagnosis.confidence,
        )

        try:
            messages = [
                SystemMessage(content=reflection_prompt),
                HumanMessage(content="Evaluate the analysis quality and completeness."),
            ]

            response = await self.llm.ainvoke(messages)
            raw_reflection = response.content.strip()

            # Parse JSON response
            reflection = self._parse_reflection_response(raw_reflection)
            logger.info(
                "Reflection result: is_complete=%s, confidence_adj=%.2f",
                reflection.get("is_complete", False),
                reflection.get("confidence_adjustment", 0),
            )
            return reflection

        except Exception as exc:
            logger.warning("Reflection LLM call failed: %s", exc)
            # Default to accepting the answer if reflection fails
            return {
                "is_complete": True,
                "missing_information": [],
                "weak_conclusions": [],
                "confidence_adjustment": 0,
                "reasoning": "Reflection skipped due to error",
                "suggested_followup": None,
            }

    def _parse_reflection_response(self, raw_response: str) -> dict[str, Any]:
        """Parse the reflection LLM's JSON response."""
        # Clean up response - remove markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        try:
            result = json.loads(cleaned)
            # Validate expected fields
            return {
                "is_complete": result.get("is_complete", True),
                "missing_information": result.get("missing_information", []),
                "weak_conclusions": result.get("weak_conclusions", []),
                "confidence_adjustment": max(
                    -0.3, min(0.3, result.get("confidence_adjustment", 0))
                ),
                "reasoning": result.get("reasoning", ""),
                "suggested_followup": result.get("suggested_followup"),
            }
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse reflection response as JSON: %s", exc)
            return {
                "is_complete": True,
                "missing_information": [],
                "weak_conclusions": [],
                "confidence_adjustment": 0,
                "reasoning": "Failed to parse reflection",
                "suggested_followup": None,
            }

    def apply_reflection(self, state: GraphState, reflection: dict[str, Any]) -> None:
        """
        Apply reflection results to the state, adjusting confidence and adding warnings.
        """
        diagnosis = state.get("diagnosis")
        if not diagnosis:
            return

        # Adjust confidence based on reflection
        confidence_adj = reflection.get("confidence_adjustment", 0)
        new_confidence = max(0.1, min(0.99, diagnosis.confidence + confidence_adj))
        diagnosis.confidence = new_confidence
        state["diagnosis"] = diagnosis

        # Add reflection warnings/notes
        if "system_warnings" not in state:
            state["system_warnings"] = []

        missing = reflection.get("missing_information", [])
        if missing:
            for item in missing[:3]:  # Limit to 3 missing items
                state["system_warnings"].append(f"[Reflection] Missing: {item}")

        weak = reflection.get("weak_conclusions", [])
        if weak:
            for item in weak[:2]:  # Limit to 2 weak conclusions
                state["system_warnings"].append(f"[Reflection] Weak conclusion: {item}")

        # Store reflection metadata for diagnostics
        state["_reflection"] = reflection

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
