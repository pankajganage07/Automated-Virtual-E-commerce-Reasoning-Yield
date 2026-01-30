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

    # Re-planning control fields
    replan_count: int  # Number of times we've re-planned (starts at 0)
    max_replans: int  # Maximum allowed re-plans (default: 2)
    needs_replan: bool  # Flag set by evaluation node
    replan_reason: str | None  # Why re-planning was triggered

    # Cannot-handle routing fields (for slimmed agent architecture)
    cannot_handle_agents: list[dict[str, Any]]  # Agents that returned cannot_handle
    route_to_analyst: bool  # Flag to route to data_analyst for complex queries

    # HITL checkpointing fields
    thread_id: str | None  # Unique ID for this conversation/session
    hitl_pending_ids: list[int]  # IDs of pending actions awaiting approval
    hitl_approved_ids: list[int]  # IDs of actions that were approved
    hitl_rejected_ids: list[int]  # IDs of actions that were rejected
    hitl_resumed: bool  # Whether this is a resumed execution

    # Internal fields for final output
    _final_answer: str
    _diagnostics: list[str]
