from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import select

from mcp_server.tools.base import BaseTool
from db.models import AgentMemory
from utils.embeddings import EmbeddingProvider  # reuse existing helper
from config import get_settings

settings = get_settings()
embedder = EmbeddingProvider(settings)


class QueryMemoryPayload(BaseModel):
    query: str
    k: int = Field(default=3, ge=1, le=10)


class QueryMemoryTool(BaseTool):
    name = "query_vector_memory"

    def request_model(self) -> type[BaseModel]:
        return QueryMemoryPayload

    async def run(self, session, payload: QueryMemoryPayload) -> dict[str, any]:
        embedding = await embedder.embed(payload.query)
        stmt = (
            select(
                AgentMemory,
                AgentMemory.embedding.l2_distance(AgentMemory.embedding, embedding).label(
                    "distance"
                ),
            )
            .order_by("distance")
            .limit(payload.k)
        )
        result = await session.execute(stmt)
        matches = []
        for record, distance in result:
            matches.append(
                {
                    "id": record.id,
                    "incident_summary": record.incident_summary,
                    "root_cause": record.root_cause,
                    "action_taken": record.action_taken,
                    "outcome": record.outcome,
                    "score": float(max(0.0, 1 / (1 + distance))) if distance is not None else 0.0,
                }
            )
        return {"matches": matches}


class SaveMemoryPayload(BaseModel):
    incident_summary: str
    root_cause: str | None = None
    action_taken: str | None = None
    outcome: str | None = None


class SaveMemoryTool(BaseTool):
    name = "save_to_memory"

    def request_model(self) -> type[BaseModel]:
        return SaveMemoryPayload

    async def run(self, session, payload: SaveMemoryPayload) -> dict[str, any]:
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
        return {"memory_id": record.id, "message": "Incident stored."}
