from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from config import Settings
from opsbrain_graph.tools.mcp_client import MCPClient


@dataclass
class MemoryIncident:
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None


@dataclass
class MemoryHit:
    id: int
    incident_summary: str
    root_cause: str | None
    action_taken: str | None
    outcome: str | None
    score: float
    created_at: dt.datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d


class MemoryService:
    """
    Memory service that routes all operations through MCP.

    This consolidates memory operations to go through the MCP server,
    ensuring consistent access patterns and centralized data management.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mcp_base_url = f"http://localhost:{settings.mcp_server_port}"
        self._mcp_api_key = settings.mcp_api_key

    async def query_similar_incidents(
        self, query: str, k: int = 3, min_score: float = 0.0
    ) -> list[MemoryHit]:
        """
        Query for similar past incidents using semantic search via MCP.

        Args:
            query: Search query for semantic similarity
            k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of matching incidents sorted by similarity
        """
        async with MCPClient(base_url=self._mcp_base_url, api_key=self._mcp_api_key) as client:
            result = await client.invoke(
                "query_vector_memory", {"query": query, "k": k, "min_score": min_score}
            )

        hits: list[MemoryHit] = []
        for match in result.get("matches", []):
            # Parse created_at if present
            created_at = None
            if match.get("created_at"):
                try:
                    created_at = dt.datetime.fromisoformat(match["created_at"])
                except (ValueError, TypeError):
                    pass

            hits.append(
                MemoryHit(
                    id=match.get("id", 0),
                    incident_summary=match.get("incident_summary", ""),
                    root_cause=match.get("root_cause"),
                    action_taken=match.get("action_taken"),
                    outcome=match.get("outcome"),
                    score=match.get("score", 0.0),
                    created_at=created_at,
                )
            )
        return hits

    async def save_incident(self, incident: MemoryIncident) -> int:
        """
        Save a new incident to memory via MCP.

        Args:
            incident: The incident details to store

        Returns:
            The ID of the newly created memory record
        """
        async with MCPClient(base_url=self._mcp_base_url, api_key=self._mcp_api_key) as client:
            result = await client.invoke(
                "save_to_memory",
                {
                    "incident_summary": incident.incident_summary,
                    "root_cause": incident.root_cause,
                    "action_taken": incident.action_taken,
                    "outcome": incident.outcome,
                },
            )

        return result.get("memory_id", 0)

    async def list_recent_incidents(
        self, limit: int = 10, offset: int = 0
    ) -> tuple[list[MemoryHit], int]:
        """
        List recent incidents from memory (without semantic search).

        Args:
            limit: Number of incidents to return
            offset: Pagination offset

        Returns:
            Tuple of (list of incidents, total count)
        """
        async with MCPClient(base_url=self._mcp_base_url, api_key=self._mcp_api_key) as client:
            result = await client.invoke("list_incidents", {"limit": limit, "offset": offset})

        hits: list[MemoryHit] = []
        for inc in result.get("incidents", []):
            created_at = None
            if inc.get("created_at"):
                try:
                    created_at = dt.datetime.fromisoformat(inc["created_at"])
                except (ValueError, TypeError):
                    pass

            hits.append(
                MemoryHit(
                    id=inc.get("id", 0),
                    incident_summary=inc.get("incident_summary", ""),
                    root_cause=inc.get("root_cause"),
                    action_taken=inc.get("action_taken"),
                    outcome=inc.get("outcome"),
                    score=1.0,  # Not from search, so no score
                    created_at=created_at,
                )
            )

        total = result.get("total", len(hits))
        return hits, total
