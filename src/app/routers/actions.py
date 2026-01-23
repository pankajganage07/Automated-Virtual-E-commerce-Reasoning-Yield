from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.schemas.actions import (
    ApproveActionRequest,
    ApproveActionResponse,
    PendingActionsResponse,
)
from app.services.hitl import PendingActionService
from app.dependencies import provide_hitl_service

router = APIRouter(prefix="/actions", tags=["Actions"])


@router.get(
    "/pending", response_model=PendingActionsResponse, status_code=status.HTTP_200_OK
)
async def list_pending_actions(
    service: PendingActionService = Depends(provide_hitl_service),
) -> PendingActionsResponse:
    return PendingActionsResponse(items=await service.list_pending())


@router.post(
    "/approve/{action_id}",
    response_model=ApproveActionResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_action(
    action_id: UUID,
    payload: ApproveActionRequest,
    service: PendingActionService = Depends(provide_hitl_service),
) -> ApproveActionResponse:
    result = await service.update_status(action_id, payload.status, payload.comment)
    return result
