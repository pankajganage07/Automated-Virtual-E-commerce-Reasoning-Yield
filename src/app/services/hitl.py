from __future__ import annotations

import datetime as dt
from typing import Iterable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from config import Settings
from db.models import PendingAction as PendingActionModel, PendingActionStatus
from db.session import async_session_factory
from langgraph.state import PendingActionProposal
from app.schemas.common import PendingAction as PendingActionSchema
from app.schemas.actions import ApproveActionResponse


class PendingActionService:
    """
    Database-backed HITL workflow coordinator.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

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
