from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.query import QueryRequest, QueryResponse, ResumeQueryRequest
from app.services.orchestrator import OrchestratorService
from app.dependencies import provide_orchestrator

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_query(
    payload: QueryRequest,
    orchestrator: OrchestratorService = Depends(provide_orchestrator),
) -> QueryResponse:
    """Execute a natural language query against the AI operations brain."""
    try:
        return await orchestrator.run_query(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestrator failure: {exc}") from exc


@router.post("/resume", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def resume_query(
    payload: ResumeQueryRequest,
    orchestrator: OrchestratorService = Depends(provide_orchestrator),
) -> QueryResponse:
    """
    Resume a paused query after HITL approval/rejection decisions.

    Use this endpoint when a previous query returned `hitl_waiting=True`.
    Provide the `thread_id` from the original response along with the
    lists of approved and rejected action IDs.
    """
    try:
        return await orchestrator.resume_query(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume failure: {exc}") from exc
