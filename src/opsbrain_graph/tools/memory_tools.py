from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class QueryVectorMemoryRequest(BaseModel):
    query: str
    k: int = 3
    filters: dict[str, Any] | None = None


class MemoryHit(BaseModel):
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None
    score: float


class QueryVectorMemoryResponse(BaseModel):
    results: list[MemoryHit]


class SaveMemoryRequest(BaseModel):
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None
    metadata: dict[str, Any] | None = None


class SaveMemoryResponse(BaseModel):
    memory_id: int
    message: str


class MemoryToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def query_memory(self, payload: QueryVectorMemoryRequest) -> QueryVectorMemoryResponse:
        result = await self._client.invoke("query_vector_memory", payload.model_dump())
        try:
            return QueryVectorMemoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for query_vector_memory: {exc}") from exc

    async def save_memory(self, payload: SaveMemoryRequest) -> SaveMemoryResponse:
        result = await self._client.invoke("save_to_memory", payload.model_dump())
        try:
            return SaveMemoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for save_to_memory: {exc}") from exc
