from config import Settings
from opsbrain_graph.tools import ToolRegistry
from opsbrain_graph.graph import OperationsGraph
from app.schemas.query import QueryRequest, QueryResponse, ResumeQueryRequest
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

    async def resume_query(self, payload: ResumeQueryRequest) -> QueryResponse:
        """
        Resume a paused query after human approval/rejection.

        Takes execution results from HITL actions and passes them to the graph
        for re-synthesis into a comprehensive answer.
        """
        supervisor_output = await self._graph.resume(
            thread_id=payload.thread_id,
            approved_action_ids=payload.approved_action_ids,
            rejected_action_ids=payload.rejected_action_ids,
            execution_results=payload.execution_results,
        )

        pending_actions = await self._hitl_service.list_pending()

        return QueryResponse(
            answer=supervisor_output.answer,
            diagnostics=supervisor_output.diagnostics,
            pending_actions=pending_actions,
            thread_id=payload.thread_id,
            hitl_waiting=False,
        )
