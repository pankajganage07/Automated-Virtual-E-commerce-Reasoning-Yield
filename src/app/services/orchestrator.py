import datetime as dt
from typing import TYPE_CHECKING

from fastapi import HTTPException

from config import Settings
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.common import PendingAction
from .hitl import PendingActionService

if TYPE_CHECKING:
    from uuid import UUID


class OrchestratorService:
    """
    Thin facade that will later call into LangGraph. For now, responds with a placeholder.
    """

    def __init__(self, settings: Settings, hitl_service: PendingActionService) -> None:
        self._settings = settings
        self._hitl = hitl_service

    async def run_query(self, payload: QueryRequest) -> QueryResponse:
        # Placeholder example diagnostics
        diagnostics = [
            f"Environment: {self._settings.environment}",
            "LangGraph orchestrator not yet implemented.",
        ]

        pending_actions = await self._hitl.list_pending()

        return QueryResponse(
            answer=(
                "Thanks for your question! The agentic workflow is not wired yet, "
                "but your query was accepted."
            ),
            diagnostics=diagnostics,
            pending_actions=pending_actions,
        )
