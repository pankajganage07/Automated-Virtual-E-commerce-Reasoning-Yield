"""
History router for incident retrieval from vector memory.

Provides endpoints for accessing past incidents and lessons learned.
"""

from fastapi import APIRouter, Depends, Query, status

from app.schemas.history import (
    IncidentListResponse,
    IncidentSearchResponse,
    IncidentItem,
)
from app.services.memory import MemoryService
from app.dependencies import provide_settings
from config import Settings

router = APIRouter(prefix="/history", tags=["History"])


def provide_memory_service(settings: Settings = Depends(provide_settings)) -> MemoryService:
    """Dependency provider for MemoryService."""
    return MemoryService(settings)


@router.get(
    "/incidents",
    response_model=IncidentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List past incidents",
    description="Retrieves past incidents and summaries from the vector DB with pagination.",
)
async def list_incidents(
    limit: int = Query(default=10, ge=1, le=50, description="Number of incidents to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    memory_service: MemoryService = Depends(provide_memory_service),
) -> IncidentListResponse:
    """
    List recent incidents from memory.

    Returns paginated list of past incidents stored in the vector database.
    """
    incidents, total = await memory_service.list_recent_incidents(limit=limit, offset=offset)

    items = [
        IncidentItem(
            id=inc.get("id"),
            incident_summary=inc.get("incident_summary", ""),
            root_cause=inc.get("root_cause"),
            action_taken=inc.get("action_taken"),
            outcome=inc.get("outcome"),
            created_at=inc.get("created_at"),
        )
        for inc in incidents
    ]

    return IncidentListResponse(
        incidents=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/incidents/search",
    response_model=IncidentSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search similar incidents",
    description="Search for similar past incidents using semantic similarity.",
)
async def search_incidents(
    query: str = Query(..., min_length=3, description="Search query for semantic similarity"),
    k: int = Query(default=5, ge=1, le=10, description="Number of results to return"),
    memory_service: MemoryService = Depends(provide_memory_service),
) -> IncidentSearchResponse:
    """
    Search for similar past incidents.

    Uses vector similarity search to find incidents matching the query.
    """
    results = await memory_service.fetch_similar_incidents(query=query, k=k)

    items = [
        IncidentItem(
            id=inc.get("id"),
            incident_summary=inc.get("incident_summary", ""),
            root_cause=inc.get("root_cause"),
            action_taken=inc.get("action_taken"),
            outcome=inc.get("outcome"),
            score=inc.get("score"),
            created_at=inc.get("created_at"),
        )
        for inc in results
    ]

    return IncidentSearchResponse(
        query=query,
        results=items,
        total_found=len(items),
    )
