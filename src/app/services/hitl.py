from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from config import Settings
from db.models import PendingAction as PendingActionModel, PendingActionStatus
from db.session import async_session_factory
from opsbrain_graph.state import PendingActionProposal
from app.schemas.common import PendingAction as PendingActionSchema
from app.schemas.actions import ApproveActionResponse, ExecuteActionResponse
from app.services.action_executor import ActionExecutor, ActionExecutionError

logger = logging.getLogger("app.hitl")


class PendingActionService:
    """
    Database-backed HITL workflow coordinator.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = ActionExecutor(settings)

    async def list_pending(self) -> list[PendingActionSchema]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(PendingActionModel)
                .where(PendingActionModel.status == PendingActionStatus.PENDING.value)
                .order_by(PendingActionModel.created_at.asc())
            )
            rows = result.scalars().all()
            return [self._to_schema(row) for row in rows]

    async def create_from_proposals(
        self,
        proposals: Sequence[PendingActionProposal],
    ) -> list[PendingActionSchema]:
        if not proposals:
            return []

        async with async_session_factory() as session:
            created: list[PendingActionModel] = []

            for proposal in proposals:
                status = (
                    PendingActionStatus.PENDING.value
                    if proposal.requires_approval
                    else PendingActionStatus.APPROVED.value
                )
                record = PendingActionModel(
                    agent_name=proposal.agent_name,
                    action_type=proposal.action_type,
                    payload=proposal.payload,
                    reasoning=proposal.reasoning,
                    status=status,
                )
                session.add(record)
                await session.flush()
                created.append(record)

            await session.commit()
            return [self._to_schema(row) for row in created]

    async def update_status(
        self,
        action_id: int,
        status: Literal["approved", "rejected"],
        comment: str | None = None,
    ) -> ApproveActionResponse:
        now = dt.datetime.utcnow()

        async with async_session_factory() as session:
            result = await session.execute(
                select(PendingActionModel)
                .where(PendingActionModel.id == action_id)
                .with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise NoResultFound(f"Pending action {action_id} not found")

            row.status = status
            row.updated_at = now
            await session.commit()

        message = comment or f"Action {status} at {now.isoformat()}."
        return ApproveActionResponse(action_id=action_id, status=status, message=message)

    async def execute_action(
        self,
        action_id: int,
    ) -> ExecuteActionResponse:
        """
        Execute an approved action by calling the corresponding MCP tool.

        Only actions with status 'approved' can be executed.
        After successful execution, status changes to 'executed'.
        """
        logger.info("=== EXECUTE_ACTION START === action_id=%s", action_id)
        now = dt.datetime.utcnow()

        async with async_session_factory() as session:
            result = await session.execute(
                select(PendingActionModel)
                .where(PendingActionModel.id == action_id)
                .with_for_update()
            )
            row = result.scalar_one_or_none()

            if row is None:
                logger.error("Action not found: action_id=%s", action_id)
                raise NoResultFound(f"Pending action {action_id} not found")

            logger.info(
                "Found action: id=%s, type=%s, status=%s, payload=%s",
                row.id,
                row.action_type,
                row.status,
                row.payload,
            )

            if row.status != PendingActionStatus.APPROVED.value:
                logger.warning(
                    "Cannot execute action with status '%s'. Expected 'approved'.",
                    row.status,
                )
                return ExecuteActionResponse(
                    action_id=action_id,
                    action_type=row.action_type,
                    status=row.status,
                    success=False,
                    message=f"Cannot execute action with status '{row.status}'. Only 'approved' actions can be executed.",
                    result=None,
                )

            # Execute the action via MCP
            try:
                logger.info(
                    "Calling executor: action_type=%s, payload=%s",
                    row.action_type,
                    row.payload,
                )
                exec_result = await self._executor.execute(
                    action_type=row.action_type,
                    payload=row.payload,
                )
                logger.info("Executor returned: %s", exec_result)

                # Update status to executed
                row.status = PendingActionStatus.EXECUTED.value
                row.updated_at = now
                await session.commit()
                logger.info("Updated action status to 'executed'")

                return ExecuteActionResponse(
                    action_id=action_id,
                    action_type=row.action_type,
                    status=PendingActionStatus.EXECUTED.value,
                    success=True,
                    message=f"Action executed successfully at {now.isoformat()}.",
                    result=exec_result,
                )
            except ActionExecutionError as exc:
                logger.error(
                    "ActionExecutionError: action_type=%s, reason=%s, details=%s",
                    exc.action_type,
                    exc.reason,
                    exc.details,
                )
                return ExecuteActionResponse(
                    action_id=action_id,
                    action_type=row.action_type,
                    status=row.status,
                    success=False,
                    message=str(exc),
                    result={"error": exc.reason, "details": exc.details},
                )

    async def approve_and_execute(
        self,
        action_id: int,
        comment: str | None = None,
    ) -> ExecuteActionResponse:
        """
        Approve an action and immediately execute it.

        This is a convenience method that combines approve + execute in one call.
        """
        logger.info("=== APPROVE_AND_EXECUTE START === action_id=%s", action_id)
        # First approve
        approve_result = await self.update_status(action_id, "approved", comment)
        logger.info("Approval result: action_id=%s, status=%s", action_id, approve_result.status)
        # Then execute
        exec_result = await self.execute_action(action_id)
        logger.info(
            "=== APPROVE_AND_EXECUTE COMPLETE === action_id=%s, success=%s, message=%s",
            action_id,
            exec_result.success,
            exec_result.message,
        )
        return exec_result

    async def list_by_status(self, statuses: Iterable[str]) -> list[PendingActionSchema]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(PendingActionModel)
                .where(PendingActionModel.status.in_(list(statuses)))
                .order_by(PendingActionModel.created_at.desc())
            )
            rows = result.scalars().all()
            return [self._to_schema(row) for row in rows]

    def _to_schema(self, row: PendingActionModel) -> PendingActionSchema:
        return PendingActionSchema(
            id=row.id,
            agent_name=row.agent_name,
            action_type=row.action_type,
            payload=row.payload,
            reasoning=row.reasoning,
            status=row.status,
        )
