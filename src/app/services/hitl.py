from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from config import Settings
from db.models import PendingAction as PendingActionModel, PendingActionStatus
from db.session import async_session_factory
from opsbrain_graph.state import PendingActionProposal
from app.schemas.common import PendingAction as PendingActionSchema
from app.schemas.actions import ApproveActionResponse, ExecuteActionResponse
from app.services.action_executor import ActionExecutor, ActionExecutionError


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

            if row.status != PendingActionStatus.APPROVED.value:
                return ExecuteActionResponse(
                    action_id=action_id,
                    status=row.status,
                    success=False,
                    message=f"Cannot execute action with status '{row.status}'. Only 'approved' actions can be executed.",
                    result=None,
                )

            # Execute the action via MCP
            try:
                exec_result = await self._executor.execute(
                    action_type=row.action_type,
                    payload=row.payload,
                )

                # Update status to executed
                row.status = PendingActionStatus.EXECUTED.value
                row.updated_at = now
                await session.commit()

                return ExecuteActionResponse(
                    action_id=action_id,
                    status=PendingActionStatus.EXECUTED.value,
                    success=True,
                    message=f"Action executed successfully at {now.isoformat()}.",
                    result=exec_result,
                )
            except ActionExecutionError as exc:
                return ExecuteActionResponse(
                    action_id=action_id,
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
        # First approve
        await self.update_status(action_id, "approved", comment)
        # Then execute
        return await self.execute_action(action_id)

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
