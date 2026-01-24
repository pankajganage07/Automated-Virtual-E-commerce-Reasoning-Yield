from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.query import QueryRequest, QueryResponse
from app.services.orchestrator import OrchestratorService
from app.dependencies import provide_orchestrator

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_query(
    payload: QueryRequest,
    orchestrator: OrchestratorService = Depends(provide_orchestrator),
) -> QueryResponse:
    try:
        return await orchestrator.run_query(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestrator failure: {exc}") from exc
