from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from .common import PendingAction


class PendingActionsResponse(BaseModel):
    items: list[PendingAction] = Field(default_factory=list)


class ApproveActionRequest(BaseModel):
    status: Literal["approved", "rejected"] = "approved"
    comment: str | None = None


class ApproveActionResponse(BaseModel):
    action_id: UUID
    status: Literal["approved", "rejected"]
    message: str
