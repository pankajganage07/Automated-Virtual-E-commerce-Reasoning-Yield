from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PendingAction(BaseModel):
    id: int
    agent_name: str = Field(..., serialization_alias="agent")
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    status: Literal["pending", "approved", "rejected", "executed"]
