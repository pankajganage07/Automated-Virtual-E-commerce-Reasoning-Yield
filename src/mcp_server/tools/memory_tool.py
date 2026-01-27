from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select, func

from mcp_server.tools.base import BaseTool
from db.models import AgentMemory
from utils.embeddings import EmbeddingProvider
from config import get_settings

settings = get_settings()
embedder = EmbeddingProvider(settings)


# =============================================================================
# QUERY VECTOR MEMORY
# =============================================================================


class QueryMemoryPayload(BaseModel):
    query: str = Field(..., description="Search query for semantic similarity")
    k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )


class QueryMemoryTool(BaseTool):
    """
    Query the vector memory for similar past incidents.

    Uses cosine distance for semantic similarity matching.
    """

    name = "query_vector_memory"

    def request_model(self) -> type[BaseModel]:
        return QueryMemoryPayload

    async def run(self, session, payload: QueryMemoryPayload) -> dict[str, Any]:
        embedding = await embedder.embed(payload.query)

        # Use cosine distance for normalized similarity (better for text embeddings)
        # cosine_distance returns 0 for identical vectors, 2 for opposite vectors
        stmt = (
            select(
                AgentMemory,
                AgentMemory.embedding.cosine_distance(embedding).label("distance"),
            )
            .order_by("distance")
            .limit(payload.k)
        )
        result = await session.execute(stmt)

        matches = []
        for record, distance in result:
            # Convert cosine distance to similarity score (1 = identical, 0 = orthogonal)
            # cosine_distance = 1 - cosine_similarity, so similarity = 1 - distance
            score = float(max(0.0, 1 - distance)) if distance is not None else 0.0

            # Apply minimum score filter
            if score >= payload.min_score:
                matches.append(
                    {
                        "id": record.id,
                        "incident_summary": record.incident_summary,
                        "root_cause": record.root_cause,
                        "action_taken": record.action_taken,
                        "outcome": record.outcome,
                        "score": round(score, 4),
                        "created_at": record.created_at.isoformat() if record.created_at else None,
                    }
                )

        return {
            "query": payload.query,
            "matches": matches,
            "total_found": len(matches),
        }


# =============================================================================
# SAVE TO MEMORY
# =============================================================================


class SaveMemoryPayload(BaseModel):
    incident_summary: str = Field(..., description="Summary of the incident")
    root_cause: str | None = Field(None, description="Identified root cause")
    action_taken: str | None = Field(None, description="Actions that were taken")
    outcome: str | None = Field(None, description="Result of the actions")


class SaveMemoryTool(BaseTool):
    """
    Save a new incident to the vector memory store.

    Generates an embedding for semantic retrieval.
    """

    name = "save_to_memory"

    def request_model(self) -> type[BaseModel]:
        return SaveMemoryPayload

    async def run(self, session, payload: SaveMemoryPayload) -> dict[str, Any]:
        embedding = await embedder.embed(payload.incident_summary)

        record = AgentMemory(
            incident_summary=payload.incident_summary,
            root_cause=payload.root_cause,
            action_taken=payload.action_taken,
            outcome=payload.outcome,
            embedding=embedding,
        )
        session.add(record)
        await session.flush()
        await session.commit()

        return {
            "memory_id": record.id,
            "message": "Incident stored successfully.",
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }


# =============================================================================
# LIST RECENT INCIDENTS
# =============================================================================


class ListIncidentsPayload(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="Number of recent incidents")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ListIncidentsTool(BaseTool):
    """
    List recent incidents from memory (without semantic search).
    """

    name = "list_incidents"

    def request_model(self) -> type[BaseModel]:
        return ListIncidentsPayload

    async def run(self, session, payload: ListIncidentsPayload) -> dict[str, Any]:
        stmt = (
            select(AgentMemory)
            .order_by(AgentMemory.created_at.desc())
            .offset(payload.offset)
            .limit(payload.limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()

        # Get total count
        count_stmt = select(func.count()).select_from(AgentMemory)
        count_result = await session.execute(count_stmt)
        total = count_result.scalar() or 0

        incidents = [
            {
                "id": record.id,
                "incident_summary": record.incident_summary,
                "root_cause": record.root_cause,
                "action_taken": record.action_taken,
                "outcome": record.outcome,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            }
            for record in records
        ]

        return {
            "incidents": incidents,
            "total": total,
            "limit": payload.limit,
            "offset": payload.offset,
        }
