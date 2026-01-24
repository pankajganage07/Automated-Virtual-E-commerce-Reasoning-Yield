from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from sqlalchemy import select

from config import Settings
from db.models import AgentMemory
from db.session import async_session_factory
from utils.embeddings import EmbeddingProvider


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
    created_at: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = EmbeddingProvider(settings)

    async def query_similar_incidents(self, query: str, k: int = 3) -> list[MemoryHit]:
        embedding = await self._embedder.embed(query)

        async with async_session_factory() as session:
            stmt = (
                select(
                    AgentMemory,
                    AgentMemory.embedding.l2_distance(AgentMemory.embedding, embedding).label(
                        "distance"
                    ),
                )
                .order_by("distance")
                .limit(k)
            )
            result = await session.execute(stmt)
            rows = result.all()

        hits: list[MemoryHit] = []
        for record, distance in rows:
            score = float(max(0.0, 1 / (1 + distance))) if distance is not None else 0.0
            hits.append(
                MemoryHit(
                    id=record.id,
                    incident_summary=record.incident_summary,
                    root_cause=record.root_cause,
                    action_taken=record.action_taken,
                    outcome=record.outcome,
                    score=score,
                    created_at=record.created_at,
                )
            )
        return hits

    async def save_incident(self, incident: MemoryIncident) -> int:
        embedding = await self._embedder.embed(incident.incident_summary)

        async with async_session_factory() as session:
            record = AgentMemory(
                incident_summary=incident.incident_summary,
                root_cause=incident.root_cause,
                action_taken=incident.action_taken,
                outcome=incident.outcome,
                embedding=embedding,
            )
            session.add(record)
            await session.flush()
            new_id = record.id
            await session.commit()

        return new_id
