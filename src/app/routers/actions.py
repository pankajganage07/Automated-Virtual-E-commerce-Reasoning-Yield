from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import NoResultFound

from app.schemas.actions import (
    ApproveActionRequest,
    ApproveActionResponse,
    ExecuteActionResponse,
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
    response_model=ApproveActionResponse | ExecuteActionResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_action(
    action_id: int,
    payload: ApproveActionRequest,
    service: PendingActionService = Depends(provide_hitl_service),
) -> ApproveActionResponse | ExecuteActionResponse:
    """
    Approve or reject a pending action.

    If `execute_immediately=True` and `status=approved`, the action will be
    executed right away and an ExecuteActionResponse is returned.
    """
    try:
        if payload.execute_immediately and payload.status == "approved":
            return await service.approve_and_execute(action_id, payload.comment)
        return await service.update_status(action_id, payload.status, payload.comment)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/execute/{action_id}",
    response_model=ExecuteActionResponse,
    status_code=status.HTTP_200_OK,
)
async def execute_action(
    action_id: int,
    service: PendingActionService = Depends(provide_hitl_service),
) -> ExecuteActionResponse:
    """
    Execute an approved action.

    The action must be in 'approved' status. After execution, the status
    changes to 'executed'.
    """
    try:
        return await service.execute_action(action_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
