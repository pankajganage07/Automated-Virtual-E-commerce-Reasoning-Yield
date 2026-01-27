from __future__ import annotations

import logging
from typing import Any, Sequence

from langgraph.graph import StateGraph, START, END

from config import Settings
from opsbrain_graph.agents import (
    AgentRunContext,
    BaseAgent,
    SalesAgent,
    InventoryAgent,
    MarketingAgent,
    SupportAgent,
    DataAnalystAgent,
    HistorianAgent,
)
from opsbrain_graph.state import GraphState
from opsbrain_graph.memory import MemoryService, MemoryIncident
from opsbrain_graph.supervisor import Supervisor, SupervisorOutput
from opsbrain_graph.tools import ToolRegistry

logger = logging.getLogger("opsbrain_graph")


class OperationsGraph:
    def __init__(self, settings: Settings, tools: ToolRegistry) -> None:
        self._settings = settings
        self._tools = tools
        self._supervisor = Supervisor(settings)
        self._memory_service = MemoryService(settings)

        self._agents: dict[str, BaseAgent] = {
            "sales": SalesAgent(tools, settings, memory_service=self._memory_service),
            "inventory": InventoryAgent(tools, settings, memory_service=self._memory_service),
            "marketing": MarketingAgent(tools, settings, memory_service=self._memory_service),
            "support": SupportAgent(tools, settings, memory_service=self._memory_service),
            "data_analyst": DataAnalystAgent(tools, settings, memory_service=self._memory_service),
            "historian": HistorianAgent(tools, settings, memory_service=self._memory_service),
        }

        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("run_tasks", self._run_tasks_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("hitl_gate", self._hitl_gate_node)
        graph.add_node("record_memory", self._record_memory_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "run_tasks")
        graph.add_edge("run_tasks", "synthesize")
        graph.add_edge("synthesize", "hitl_gate")
        graph.add_edge("hitl_gate", "record_memory")
        graph.add_edge("record_memory", END)

        return graph.compile()

    async def _plan_node(self, state: GraphState) -> GraphState:
        logger.info("Planning tasks for query: %s", state["user_query"])
        tasks = self._supervisor.plan(state)
        state["battle_plan"] = tasks
        logger.info("Battle plan: %s", [t.agent for t in tasks])
        return state

    async def _run_tasks_node(self, state: GraphState) -> GraphState:
        battle_plan = state.get("battle_plan", [])
        context = AgentRunContext(
            user_query=state["user_query"],
            conversation_history=state.get("conversation_history", []),
            metadata=state.get("metadata", {}),
        )

        for task in battle_plan:
            agent = self._agents.get(task.agent)
            if not agent:
                logger.warning("Unknown agent: %s", task.agent)
                continue

            logger.info("Running agent: %s", task.agent)
            try:
                result = await agent.run(task, context)
                self._supervisor.incorporate_agent_result(state, task.agent, result)
                logger.info("Agent %s completed with status: %s", task.agent, result.status)
            except Exception as exc:
                logger.exception("Agent %s failed: %s", task.agent, exc)
                if "system_warnings" not in state:
                    state["system_warnings"] = []
                state["system_warnings"].append(f"Agent {task.agent} crashed: {exc}")

        return state

    async def _synthesize_node(self, state: GraphState) -> GraphState:
        logger.info("Synthesizing results...")
        output = await self._supervisor.synthesize(state)  # Now async!
        state["diagnosis"] = output.summary
        state["_final_answer"] = output.answer
        state["_diagnostics"] = output.diagnostics
        state["pending_action_proposals"] = output.pending_actions
        return state

    async def _hitl_gate_node(self, state: GraphState) -> GraphState:
        proposals = state.get("pending_action_proposals", [])
        state["hitl_wait"] = bool([p for p in proposals if p.requires_approval])
        return state

    async def _record_memory_node(self, state: GraphState) -> GraphState:
        diagnosis = state.get("diagnosis")
        if diagnosis and diagnosis.confidence > 0.7:
            try:
                incident = MemoryIncident(
                    incident_summary=state["user_query"],
                    root_cause=diagnosis.narrative[:500] if diagnosis.narrative else None,
                    action_taken=None,
                    outcome=None,
                )
                memory_id = await self._memory_service.save_incident(incident)
                logger.info("Recorded memory with ID: %s", memory_id)
            except Exception as exc:
                logger.warning("Failed to record memory: %s", exc)
        return state

    async def run(
        self,
        query: str,
        conversation_history: Sequence[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SupervisorOutput:
        initial_state = self._supervisor.initialize_state(query, conversation_history)
        if metadata:
            initial_state["metadata"] = metadata

        final_state = await self._graph.ainvoke(initial_state)

        return SupervisorOutput(
            summary=final_state.get("diagnosis"),
            answer=final_state.get("_final_answer", "No answer generated."),
            diagnostics=final_state.get("_diagnostics", []),
            pending_actions=final_state.get("pending_action_proposals", []),
        )
