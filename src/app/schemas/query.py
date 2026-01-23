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
