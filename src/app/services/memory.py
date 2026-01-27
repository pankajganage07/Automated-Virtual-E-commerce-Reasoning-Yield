"""
Memory Service for the App layer.

This is a thin wrapper that delegates to the opsbrain_graph.memory module,
which in turn routes all operations through MCP.
"""

from config import Settings
from opsbrain_graph.memory import MemoryService as OpsBrainMemoryService, MemoryHit, MemoryIncident


class MemoryService:
    """
    App-layer memory service that delegates to the MCP-based implementation.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._memory = OpsBrainMemoryService(settings)

    async def fetch_similar_incidents(self, query: str, k: int = 3) -> list[dict]:
        """Fetch similar past incidents using semantic search."""
        hits = await self._memory.query_similar_incidents(query, k)
        return [hit.to_dict() for hit in hits]

    async def save_incident(
        self,
        summary: str,
        root_cause: str | None = None,
        action_taken: str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Save a new incident to memory."""
        incident = MemoryIncident(
            incident_summary=summary,
            root_cause=root_cause,
            action_taken=action_taken,
            outcome=outcome,
        )
        return await self._memory.save_incident(incident)

    async def list_recent_incidents(
        self, limit: int = 10, offset: int = 0
    ) -> tuple[list[dict], int]:
        """List recent incidents with pagination."""
        hits, total = await self._memory.list_recent_incidents(limit, offset)
        return [hit.to_dict() for hit in hits], total
