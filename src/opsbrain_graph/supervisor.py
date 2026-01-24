from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from config import Settings
from opsbrain_graph.agents import AgentResult, AgentTask, AgentRecommendation
from opsbrain_graph.state import DiagnosisSummary, GraphState, PendingActionProposal


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

    def initialize_state(
        self,
        user_query: str,
        conversation_history: Sequence[dict[str, Any]] | None = None,
    ) -> GraphState:
        return GraphState(
            user_query=user_query,
            conversation_history=list(conversation_history or []),
        )

    def plan(self, state: GraphState) -> list[AgentTask]:
        query = state.user_query.lower()

        tasks: list[AgentTask] = []

        if any(keyword in query for keyword in ("sale", "revenue", "drop", "trend")):
            tasks.append(
                AgentTask(
                    agent="sales",
                    objective="Analyze revenue trends and detect anomalies.",
                    parameters={"window_days": 7, "group_by": "day"},
                    result_slot="agent_findings.sales",
                )
            )

        if any(keyword in query for keyword in ("stock", "inventory", "out of stock", "restock")):
            tasks.append(
                AgentTask(
                    agent="inventory",
                    objective="Check stock levels for key products mentioned or top sellers.",
                    parameters={"product_ids": state.metadata.get("focus_product_ids", [1, 2, 3])},
                    result_slot="agent_findings.inventory",
                )
            )

        if any(keyword in query for keyword in ("campaign", "ad", "marketing", "roas")):
            tasks.append(
                AgentTask(
                    agent="marketing",
                    objective="Evaluate campaign spend efficiency and conflicts.",
                    parameters={"window_days": 7},
                    result_slot="agent_findings.marketing",
                )
            )

        if any(keyword in query for keyword in ("ticket", "support", "sentiment", "complaint")):
            tasks.append(
                AgentTask(
                    agent="support",
                    objective="Summarize support sentiment and spikes in issues.",
                    parameters={"window_days": 7},
                    result_slot="agent_findings.support",
                )
            )

        if "histor" in query or not tasks:
            tasks.append(
                AgentTask(
                    agent="historian",
                    objective="Retrieve similar past incidents.",
                    parameters={"mode": "query", "query": state.user_query},
                    result_slot="memory_context",
                )
            )

        state.battle_plan = tasks
        return tasks

    def incorporate_agent_result(
        self,
        state: GraphState,
        agent_name: str,
        result: AgentResult,
    ) -> None:
        if result.status == "success":
            state.record_agent_result(
                agent_name=agent_name,
                findings=result.findings,
                insights=result.insights,
                recommendations=result.recommendations,
            )
        else:
            state.add_warning(f"{agent_name} agent failed: {result.errors}")

    def synthesize(self, state: GraphState) -> SupervisorOutput:
        summary = self._build_diagnosis(state)
        recommendations = self._collect_pending_actions(state)

        answer = self._compose_answer(summary, recommendations, state.system_warnings)
        diagnostics = self._compile_diagnostics(state)

        return SupervisorOutput(
            summary=summary,
            answer=answer,
            diagnostics=diagnostics,
            pending_actions=recommendations,
        )

    def _build_diagnosis(self, state: GraphState) -> DiagnosisSummary:
        insights = []
        for agent, agent_insights in state.agent_insights.items():
            for insight in agent_insights:
                insights.append(f"{agent.title()}: {insight}")

        narrative = (
            " ".join(insights) if insights else "Investigation completed; awaiting more signals."
        )
        confidence = min(0.95, 0.5 + 0.1 * len(insights))

        summary = DiagnosisSummary(
            narrative=narrative,
            key_findings=insights,
            confidence=confidence,
        )
        state.diagnosis = summary
        return summary

    def _collect_pending_actions(self, state: GraphState) -> list[PendingActionProposal]:
        proposals: list[PendingActionProposal] = []

        for recommendation in state.recommendations:
            proposal = PendingActionProposal(
                agent_name=recommendation.action_type.split("_")[0],
                action_type=(
                    rec.item.action_type
                    if isinstance(rec := recommendation, AgentRecommendation)
                    else ""
                ),
                payload=recommendation.payload,
                reasoning=recommendation.reasoning,
                requires_approval=recommendation.requires_approval,
            )
            proposals.append(proposal)

        state.pending_action_proposals = proposals
        state.hitl_wait = bool(proposals)
        return proposals

    def _compose_answer(
        self,
        summary: DiagnosisSummary,
        proposals: list[PendingActionProposal],
        warnings: list[str],
    ) -> str:
        lines = [summary.narrative]

        if proposals:
            lines.append("\nRecommended actions awaiting approval:")
            for proposal in proposals:
                lines.append(f"- [{proposal.action_type}] {proposal.reasoning}")

        if warnings:
            lines.append("\nWarnings:")
            for warning in warnings:
                lines.append(f"- {warning}")

        return "\n".join(lines).strip()

    def _compile_diagnostics(self, state: GraphState) -> list[str]:
        diag = [f"Agents executed: {', '.join(state.agent_findings.keys()) or 'none'}"]
        if state.hitl_wait:
            diag.append("HITL pending actions detected.")
        if state.system_warnings:
            diag.append("Warnings present; see answer body.")
        return diag
