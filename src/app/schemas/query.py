from typing import Any, List

from pydantic import BaseModel, Field

from .common import PendingAction


class QueryRequest(BaseModel):
    question: str
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    diagnostics: List[str] = Field(default_factory=list)
    pending_actions: List[PendingAction] = Field(default_factory=list)
    thread_id: str | None = Field(
        default=None,
        description="Thread ID for resuming this query after HITL approval",
    )
    hitl_waiting: bool = Field(
        default=False,
        description="True if query is paused waiting for human approval",
    )


class ResumeQueryRequest(BaseModel):
    """Request to resume a paused query after HITL decisions."""

    thread_id: str = Field(..., description="Thread ID from the original query")
    approved_action_ids: List[int] = Field(
        default_factory=list,
        description="IDs of actions that were approved",
    )
    rejected_action_ids: List[int] = Field(
        default_factory=list,
        description="IDs of actions that were rejected",
    )
