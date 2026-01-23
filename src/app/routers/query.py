from fastapi import APIRouter, Depends, status

from app.schemas.query import QueryRequest, QueryResponse
from app.services.orchestrator import OrchestratorService
from app.dependencies import provide_orchestrator

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_query(
    payload: QueryRequest,
    orchestrator: OrchestratorService = Depends(provide_orchestrator),
) -> QueryResponse:
    """
    Entry point for user questions. Currently returns a stub response.
    """
    return await orchestrator.run_query(payload)
