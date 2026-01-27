from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


# =============================================================================
# QUERY VECTOR MEMORY
# =============================================================================


class QueryVectorMemoryRequest(BaseModel):
    """Request for semantic search of past incidents."""

    query: str = Field(..., description="Search query for semantic similarity")
    k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class MemoryHit(BaseModel):
    """A single matching incident from memory."""

    id: int | None = None
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None
    score: float
    created_at: str | None = None


class QueryVectorMemoryResponse(BaseModel):
    """Response from memory query."""

    query: str | None = None
    matches: list[MemoryHit]
    total_found: int | None = None


# =============================================================================
# SAVE TO MEMORY
# =============================================================================


class SaveMemoryRequest(BaseModel):
    """Request to save a new incident to memory."""

    incident_summary: str = Field(..., description="Summary of the incident")
    root_cause: str | None = Field(None, description="Identified root cause")
    action_taken: str | None = Field(None, description="Actions that were taken")
    outcome: str | None = Field(None, description="Result of the actions")


class SaveMemoryResponse(BaseModel):
    """Response from saving to memory."""

    memory_id: int
    message: str
    created_at: str | None = None


# =============================================================================
# LIST INCIDENTS
# =============================================================================


class ListIncidentsRequest(BaseModel):
    """Request to list recent incidents."""

    limit: int = Field(default=10, ge=1, le=50, description="Number of incidents to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class IncidentRecord(BaseModel):
    """An incident record from memory."""

    id: int
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None
    created_at: str | None = None


class ListIncidentsResponse(BaseModel):
    """Response from listing incidents."""

    incidents: list[IncidentRecord]
    total: int
    limit: int
    offset: int


# =============================================================================
# MEMORY TOOLSET
# =============================================================================


class MemoryToolset:
    """Toolset for memory operations via MCP."""

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def query_memory(self, payload: QueryVectorMemoryRequest) -> QueryVectorMemoryResponse:
        """Query for similar past incidents using semantic search."""
        result = await self._client.invoke("query_vector_memory", payload.model_dump())
        try:
            # Map 'matches' key from MCP response
            if "results" not in result and "matches" in result:
                result["matches"] = result.get("matches", [])
            return QueryVectorMemoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for query_vector_memory: {exc}") from exc

    async def save_memory(self, payload: SaveMemoryRequest) -> SaveMemoryResponse:
        """Save a new incident to memory."""
        result = await self._client.invoke("save_to_memory", payload.model_dump())
        try:
            return SaveMemoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for save_to_memory: {exc}") from exc

    async def list_incidents(self, payload: ListIncidentsRequest) -> ListIncidentsResponse:
        """List recent incidents (without semantic search)."""
        result = await self._client.invoke("list_incidents", payload.model_dump())
        try:
            return ListIncidentsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for list_incidents: {exc}") from exc
