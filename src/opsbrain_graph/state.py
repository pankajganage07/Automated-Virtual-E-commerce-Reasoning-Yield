from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from opsbrain_graph.agents.base_agent import AgentRecommendation, AgentTask


@dataclass
class PendingActionProposal:
    agent_name: str
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    requires_approval: bool = True


@dataclass
class DiagnosisSummary:
    narrative: str = ""
    key_findings: list[str] = field(default_factory=list)
    confidence: float = 0.5


def _default_diagnosis() -> dict[str, Any]:
    return {"narrative": "", "key_findings": [], "confidence": 0.5}


class GraphState(TypedDict, total=False):
    """
    LangGraph state using TypedDict for proper serialization.
    """

    user_query: str
    conversation_history: list[dict[str, Any]]

    battle_plan: list[AgentTask]
    agent_findings: dict[str, dict[str, Any]]
    agent_insights: dict[str, list[str]]

    structured_evidence: list[dict[str, Any]]
    diagnosis: dict[str, Any]

    recommendations: list[AgentRecommendation]
    pending_action_proposals: list[PendingActionProposal]

    memory_context: list[dict[str, Any]]
    system_warnings: list[str]

    hitl_wait: bool
    metadata: dict[str, Any]


def create_initial_state(
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> GraphState:
    """Factory function to create a properly initialized GraphState."""
    return GraphState(
        user_query=user_query,
        conversation_history=conversation_history or [],
        battle_plan=[],
        agent_findings={},
        agent_insights={},
        structured_evidence=[],
        diagnosis=_default_diagnosis(),
        recommendations=[],
        pending_action_proposals=[],
        memory_context=[],
        system_warnings=[],
        hitl_wait=False,
        metadata={},
    )


def record_agent_result(
    state: GraphState,
    agent_name: str,
    findings: dict[str, Any],
    insights: list[str],
    recommendations: list[AgentRecommendation],
) -> None:
    """Helper function to record agent results into state."""
    if agent_name == "historian" and "matches" in findings:
        state["memory_context"] = findings["matches"]

    state["agent_findings"][agent_name] = findings
    state["agent_insights"][agent_name] = insights
    state["recommendations"].extend(recommendations)


def add_warning(state: GraphState, message: str) -> None:
    """Helper function to add a warning to state."""
    state["system_warnings"].append(message)
