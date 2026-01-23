from __future__ import annotations

import asyncio
import datetime as dt
from typing import Literal
from uuid import UUID, uuid4

from config import Settings
from app.schemas.actions import ApproveActionResponse
from app.schemas.common import PendingAction


class PendingActionService:
    """
    Stub HITL service — stores actions in memory until DB integration arrives.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._store: dict[UUID, PendingAction] = {}

    async def list_pending(self) -> list[PendingAction]:
        async with self._lock:
            return [
                action for action in self._store.values() if action.status == "pending"
            ]

    async def create_stub_action(self) -> PendingAction:
        """
        Helper used during development to simulate pending actions.
        """
        async with self._lock:
            action_id = uuid4()
            action = PendingAction(
                id=action_id,
                agent_name="inventory",
                action_type="restock_item",
                payload={"product_id": 101, "qty": 50},
                reasoning="Low stock detected during pre-flight check.",
                status="pending",
            )
            self._store[action_id] = action
            return action

    async def update_status(
        self,
        action_id: UUID,
        status: Literal["approved", "rejected"],
        comment: str | None = None,
    ) -> ApproveActionResponse:
        async with self._lock:
            action = self._store.get(action_id)
            if action is None:
                raise ValueError(f"Action {action_id} not found")
            action.status = status
        return ApproveActionResponse(
            action_id=action_id,
            status=status,
            message=comment or f"Action {status} by user.",
        )
