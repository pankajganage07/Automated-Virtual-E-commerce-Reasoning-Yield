from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import PendingAction


class PendingActionsResponse(BaseModel):
    items: list[PendingAction] = Field(default_factory=list)


class ApproveActionRequest(BaseModel):
    status: Literal["approved", "rejected"] = "approved"
    comment: str | None = None
    execute_immediately: bool = Field(
        default=False,
        description="If True and status is 'approved', execute the action immediately after approval.",
    )


class ApproveActionResponse(BaseModel):
    action_id: int
    status: Literal["approved", "rejected"]
    message: str


class ExecuteActionResponse(BaseModel):
    """Response from executing an approved action."""

    action_id: int
    status: Literal["pending", "approved", "rejected", "executed"]
    success: bool
    message: str
    result: dict[str, Any] | None = None
