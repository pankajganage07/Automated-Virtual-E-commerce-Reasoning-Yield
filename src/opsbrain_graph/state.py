from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from opsbrain_graph.agents.base_agent import AgentTask, AgentRecommendation


@dataclass
class DiagnosisSummary:
    narrative: str
    key_findings: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class PendingActionProposal:
    agent_name: str
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    requires_approval: bool = True


class GraphState(TypedDict, total=False):
    """State passed through the LangGraph."""

    user_query: str
    conversation_history: list[dict[str, Any]]
    battle_plan: list[AgentTask]
    agent_findings: dict[str, dict[str, Any]]
    agent_insights: dict[str, list[str]]
    recommendations: list[AgentRecommendation]
    memory_context: list[dict[str, Any]]
    diagnosis: DiagnosisSummary | None
    pending_action_proposals: list[PendingActionProposal]
    hitl_wait: bool
    system_warnings: list[str]
    metadata: dict[str, Any]

    # Internal fields for final output
    _final_answer: str
    _diagnostics: list[str]
