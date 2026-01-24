from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import NoResultFound

from app.schemas.actions import (
    ApproveActionRequest,
    ApproveActionResponse,
    PendingActionsResponse,
)
from app.services.hitl import PendingActionService
from app.dependencies import provide_hitl_service

router = APIRouter(prefix="/actions", tags=["HITL Actions"])


@router.get("/pending", response_model=PendingActionsResponse, status_code=status.HTTP_200_OK)
async def list_pending_actions(
    service: PendingActionService = Depends(provide_hitl_service),
) -> PendingActionsResponse:
    entries = await service.list_pending()
    return PendingActionsResponse(items=entries)


@router.post(
    "/approve/{action_id}",
    response_model=ApproveActionResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_action(
    action_id: int,
    payload: ApproveActionRequest,
    service: PendingActionService = Depends(provide_hitl_service),
) -> ApproveActionResponse:
    try:
        return await service.update_status(action_id, payload.status, payload.comment)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
