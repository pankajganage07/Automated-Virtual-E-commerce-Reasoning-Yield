from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence

from langgraph.graph import StateGraph, START, END

from config import Settings
from opsbrain_graph.agents import (
    AgentRunContext,
    AgentTask,
    AgentResult,
    AgentRecommendation,
    BaseAgent,
    SalesAgent,
    InventoryAgent,
    MarketingAgent,
    SupportAgent,
    DataAnalystAgent,
    HistorianAgent,
)
from opsbrain_graph.state import GraphState, PendingActionProposal, add_warning
from opsbrain_graph.supervisor import Supervisor, SupervisorOutput
from opsbrain_graph.tools import ToolRegistry
from opsbrain_graph.memory import MemoryService, MemoryIncident


class OperationsGraph:
    """
    High-level wrapper around LangGraph state machine for the Operations Brain.
    """

    def __init__(
        self, settings: Settings, tools: ToolRegistry, memory_service: MemoryService | None = None
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.memory_service = memory_service
        self.supervisor = Supervisor(settings)
        self.agents: dict[str, BaseAgent] = {
            "sales": SalesAgent(tools, settings),
            "inventory": InventoryAgent(tools, settings),
            "marketing": MarketingAgent(tools, settings),
            "support": SupportAgent(tools, settings),
            "data_analyst": DataAnalystAgent(tools, settings),
            "historian": HistorianAgent(tools, settings, memory_service=memory_service),
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

    async def _record_memory_node(self, state: GraphState) -> GraphState:
        if not self.memory_service:
            return state
        if not state["metadata"].get("persist_incident", False):
            return state

        diagnosis = state["diagnosis"]
        summary = diagnosis.get("narrative", "").strip()
        if not summary:
            return state

        root_cause = "; ".join(diagnosis.get("key_findings", [])[:3]) or None
        actions = (
            "; ".join(
                f"{action.action_type}: {action.reasoning}"
                for action in state["pending_action_proposals"]
            )
            or None
        )
        outcome = "pending_approval" if state["hitl_wait"] else "analysis_shared"

        incident = MemoryIncident(
            incident_summary=summary,
            root_cause=root_cause,
            action_taken=actions,
            outcome=outcome,
        )
        await self.memory_service.save_incident(incident)
        return state

    async def _hitl_gate_node(self, state: GraphState) -> GraphState:
        if state["hitl_wait"]:
            state["metadata"]["wait_state"] = {
                "pending_action_count": len(state["pending_action_proposals"]),
                "note": "Awaiting human approval before execution.",
            }
        return state

    async def run(
        self,
        user_query: str,
        conversation_history: Sequence[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SupervisorOutput:
        state = self.supervisor.initialize_state(user_query, conversation_history)
        if metadata:
            state["metadata"].update(metadata)

        final_state: GraphState = await self._graph.ainvoke(state)
        return self.supervisor.synthesize(final_state)

    async def _plan_node(self, state: GraphState) -> GraphState:
        self.supervisor.plan(state)
        return state

    async def _run_tasks_node(self, state: GraphState) -> GraphState:
        tasks = state["battle_plan"]
        if not tasks:
            add_warning(state, "Supervisor produced no tasks; skipping worker execution.")
            return state

        ctx = AgentRunContext(
            user_query=state["user_query"],
            conversation_history=state["conversation_history"],
            memory_context=state["memory_context"],
            state_snapshot={
                "agent_findings": state["agent_findings"],
                "recommendations": [rec.__dict__ for rec in state["recommendations"]],
            },
        )

        task_results = await self._execute_tasks(tasks, ctx)

        for agent_name, result in task_results:
            self.supervisor.incorporate_agent_result(state, agent_name, result)

        return state

    async def _synthesize_node(self, state: GraphState) -> GraphState:
        self.supervisor.synthesize(state)
        return state

    async def _execute_tasks(
        self, tasks: Sequence[AgentTask], context: AgentRunContext
    ) -> list[tuple[str, AgentResult]]:
        coroutines = [self._run_agent_with_retry(task.agent, task, context) for task in tasks]
        return await asyncio.gather(*coroutines)

    async def _run_agent_with_retry(
        self,
        agent_name: str,
        task: AgentTask,
        context: AgentRunContext,
        max_attempts: int = 2,
        delay: float = 1.0,
    ) -> tuple[str, AgentResult]:
        agent = self.agents.get(agent_name)
        if not agent:
            return agent_name, AgentResult(
                status="failure", errors=f"Agent '{agent_name}' not registered."
            )

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = await agent.run(task, context)
                if result.status == "needs_retry" and attempt < max_attempts:
                    await asyncio.sleep(delay)
                    continue
                return agent_name, result
            except Exception as exc:  # final catch
                if attempt >= max_attempts:
                    return agent_name, agent.failure(exc)
                await asyncio.sleep(delay)

        return agent_name, agent.failure("Unknown execution failure.")
