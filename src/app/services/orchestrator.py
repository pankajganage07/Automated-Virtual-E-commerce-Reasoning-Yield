from config import Settings
from opsbrain_graph.tools import ToolRegistry
from opsbrain_graph.graph import OperationsGraph
from app.schemas.query import QueryRequest, QueryResponse
from .hitl import PendingActionService
from app.services.memory import MemoryService


class OrchestratorService:
    """
    Facade between FastAPI and the LangGraph workflow.
    """

    def __init__(self, settings: Settings, hitl_service: PendingActionService) -> None:
        self._settings = settings
        self._hitl_service = hitl_service
        self._tools = ToolRegistry.from_settings(settings)
        self._memory_service = MemoryService(settings)
        self._graph = OperationsGraph(settings, self._tools)

    async def run_query(self, payload: QueryRequest) -> QueryResponse:
        supervisor_output, thread_id, hitl_waiting = await self._graph.run(
            query=payload.question,
            conversation_history=payload.metadata.get("history") if payload.metadata else None,
            metadata=payload.metadata,
            thread_id=payload.metadata.get("thread_id") if payload.metadata else None,
        )

        if supervisor_output.pending_actions:
            await self._hitl_service.create_from_proposals(supervisor_output.pending_actions)

        pending_actions = await self._hitl_service.list_pending()

        return QueryResponse(
            answer=supervisor_output.answer,
            diagnostics=supervisor_output.diagnostics,
            pending_actions=pending_actions,
            thread_id=thread_id,
            hitl_waiting=hitl_waiting,
        )

    async def resume_query(
        self,
        thread_id: str,
        approved_action_ids: list[int] | None = None,
        rejected_action_ids: list[int] | None = None,
    ) -> QueryResponse:
        """
        Resume a paused query after human approval/rejection.
        """
        supervisor_output = await self._graph.resume(
            thread_id=thread_id,
            approved_action_ids=approved_action_ids,
            rejected_action_ids=rejected_action_ids,
        )

        pending_actions = await self._hitl_service.list_pending()

        return QueryResponse(
            answer=supervisor_output.answer,
            diagnostics=supervisor_output.diagnostics,
            pending_actions=pending_actions,
            thread_id=thread_id,
            hitl_waiting=False,
        )
