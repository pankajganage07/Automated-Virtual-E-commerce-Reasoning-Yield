from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from opsbrain_graph.agents import AgentResult, AgentTask, AgentRecommendation
from opsbrain_graph.state import GraphState, DiagnosisSummary, PendingActionProposal
from utils.llm import get_llm

logger = logging.getLogger("supervisor")


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

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = None

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
        )

    def plan(self, state: GraphState) -> list[AgentTask]:
        query = state["user_query"].lower()
        tasks: list[AgentTask] = []

        # Top products query (must check before general sales)
        if any(kw in query for kw in ("top", "best", "highest", "most sold", "best selling")):
            if any(kw in query for kw in ("product", "item", "sku", "selling")):
                limit = 5  # default
                # Try to extract number from query
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

        # Default: if no specific agent matched, use sales + support as general check
        if not tasks:
            tasks.append(
                AgentTask(
                    agent="sales",
                    objective="General sales health check.",
                    parameters={"window_days": 7, "group_by": "day"},
                    result_slot="agent_findings.sales",
                )
            )

        state["battle_plan"] = tasks
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
        else:
            if "system_warnings" not in state:
                state["system_warnings"] = []
            state["system_warnings"].append(f"{agent_name} agent failed: {result.errors}")

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
