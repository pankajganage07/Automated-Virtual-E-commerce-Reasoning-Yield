from typing import Literal

from pydantic import BaseModel, Field

from .common import PendingAction


class PendingActionsResponse(BaseModel):
    items: list[PendingAction] = Field(default_factory=list)


class ApproveActionRequest(BaseModel):
    status: Literal["approved", "rejected"] = "approved"
    comment: str | None = None


class ApproveActionResponse(BaseModel):
    action_id: int
    status: Literal["approved", "rejected"]
    message: str
