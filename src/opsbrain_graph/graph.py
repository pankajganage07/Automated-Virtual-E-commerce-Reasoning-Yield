from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
from utils.observability import (
    LangSmithConfig,
    TracingCallbackHandler,
)

logger = logging.getLogger("opsbrain_graph")


class OperationsGraph:
    def __init__(self, settings: Settings, tools: ToolRegistry) -> None:
        self._settings = settings
        self._tools = tools
        self._supervisor = Supervisor(settings)
        self._memory_service = MemoryService(settings)

        # Initialize LangSmith tracing for the graph
        self._tracing_enabled = LangSmithConfig.initialize(settings)
        if self._tracing_enabled:
            LangSmithConfig.set_project(settings.langsmith_project)
            logger.info("LangSmith tracing enabled for OperationsGraph")

        self._agents: dict[str, BaseAgent] = {
            "sales": SalesAgent(tools, settings, memory_service=self._memory_service),
            "inventory": InventoryAgent(tools, settings, memory_service=self._memory_service),
            "marketing": MarketingAgent(tools, settings, memory_service=self._memory_service),
            "support": SupportAgent(tools, settings, memory_service=self._memory_service),
            "data_analyst": DataAnalystAgent(tools, settings, memory_service=self._memory_service),
            "historian": HistorianAgent(tools, settings, memory_service=self._memory_service),
        }

        # Register agents with supervisor for dynamic prompt generation
        self._supervisor.register_agents(self._agents)

        # Initialize checkpointer for HITL state persistence
        self._checkpointer = MemorySaver()
        
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        # Define nodes
        graph.add_node("plan", self._plan_node)
        graph.add_node("run_tasks", self._run_tasks_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("replan", self._replan_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("hitl_gate", self._hitl_gate_node)
        graph.add_node("execute_approved", self._execute_approved_node)
        graph.add_node("record_memory", self._record_memory_node)

        # Define edges with conditional routing
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "run_tasks")
        graph.add_edge("run_tasks", "evaluate")
        
        # Conditional edge: evaluate results and decide next step
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "synthesize": "synthesize",
                "replan": "replan",
            }
        )
        
        # After re-planning, run the new tasks
        graph.add_edge("replan", "run_tasks")
        
        # Synthesis leads to HITL gate
        graph.add_edge("synthesize", "hitl_gate")
        
        # Conditional edge: HITL gate decides if we wait or continue
        graph.add_conditional_edges(
            "hitl_gate",
            self._route_after_hitl,
            {
                "wait": END,  # Pause for human approval (will resume later)
                "execute": "execute_approved",
                "skip": "record_memory",
            }
        )
        
        # After executing approved actions, record memory
        graph.add_edge("execute_approved", "record_memory")
        graph.add_edge("record_memory", END)

        # Compile with checkpointer for HITL state persistence
        return graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["execute_approved"],  # Interrupt before executing if HITL pending
        )

    async def _plan_node(self, state: GraphState) -> GraphState:
        logger.info("Planning tasks for query: %s", state["user_query"])
        tasks = await self._supervisor.plan(state)  # Now async with LLM-based planning
        state["battle_plan"] = tasks
        logger.info("Battle plan: %s", [t.agent for t in tasks])
        return state

    async def _run_tasks_node(self, state: GraphState) -> GraphState:
        """Execute agent tasks in parallel using asyncio.gather."""
        battle_plan = state.get("battle_plan", [])
        context = AgentRunContext(
            user_query=state["user_query"],
            conversation_history=state.get("conversation_history", []),
            metadata=state.get("metadata", {}),
        )

        # Filter valid tasks and prepare for parallel execution
        valid_tasks = []
        for task in battle_plan:
            agent = self._agents.get(task.agent)
            if not agent:
                logger.warning("Unknown agent: %s", task.agent)
                continue
            valid_tasks.append((task, agent))

        if not valid_tasks:
            logger.warning("No valid tasks to execute")
            return state

        # Execute all agents in parallel
        logger.info("Executing %d agents in parallel: %s", len(valid_tasks), [t[0].agent for t in valid_tasks])
        
        async def run_agent_safe(task, agent):
            """Run a single agent with error handling and tracing."""
            start_time = time.perf_counter()
            TracingCallbackHandler.on_agent_start(task.agent, task, context)
            try:
                result = await agent.run(task, context)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Agent %s completed with status: %s", task.agent, result.status)
                TracingCallbackHandler.on_agent_end(task.agent, result, duration_ms)
                return (task.agent, result, None)
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.exception("Agent %s failed: %s", task.agent, exc)
                TracingCallbackHandler.on_agent_end(task.agent, None, duration_ms)
                return (task.agent, None, str(exc))

        # Run all agents concurrently
        results = await asyncio.gather(
            *[run_agent_safe(task, agent) for task, agent in valid_tasks],
            return_exceptions=False,  # We handle exceptions in run_agent_safe
        )

        # Incorporate results into state
        for agent_name, result, error in results:
            if result is not None:
                self._supervisor.incorporate_agent_result(state, agent_name, result)
            elif error is not None:
                if "system_warnings" not in state:
                    state["system_warnings"] = []
                state["system_warnings"].append(f"Agent {agent_name} crashed: {error}")

        return state

    async def _evaluate_node(self, state: GraphState) -> GraphState:
        """Evaluate agent results to determine if re-planning is needed."""
        logger.info("Evaluating agent results...")
        
        # Use supervisor's evaluation logic (modifies state in-place)
        self._supervisor.evaluate_results(state)
        
        if state.get("needs_replan"):
            logger.info(
                "Re-planning triggered (attempt %d/%d): %s",
                state.get("replan_count", 0) + 1,
                state.get("max_replans", 2),
                state.get("replan_reason", "Unknown"),
            )
        else:
            logger.info("Results sufficient, proceeding to synthesis")
        
        return state

    def _route_after_evaluation(self, state: GraphState) -> str:
        """Conditional routing function: decide whether to replan or synthesize."""
        if state.get("needs_replan", False):
            return "replan"
        return "synthesize"

    async def _replan_node(self, state: GraphState) -> GraphState:
        """Re-plan with consideration for what failed."""
        logger.info("Re-planning (attempt %d)...", state.get("replan_count", 0) + 1)
        
        tasks = await self._supervisor.replan(state)
        
        if tasks:
            state["battle_plan"] = tasks
            logger.info("New battle plan: %s", [t.agent for t in tasks])
        else:
            # No new tasks, force proceed to synthesis
            state["needs_replan"] = False
            logger.warning("No new tasks from re-planning, proceeding to synthesis")
        
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
        """
        HITL gate: Check for pending actions that require human approval.
        
        This node determines whether to:
        - Wait for human approval (interrupt)
        - Execute pre-approved actions
        - Skip directly to memory recording
        """
        proposals = state.get("pending_action_proposals", [])
        requiring_approval = [p for p in proposals if p.requires_approval]
        
        state["hitl_wait"] = bool(requiring_approval)
        state["hitl_pending_ids"] = []  # Will be populated after DB save
        state["hitl_approved_ids"] = state.get("hitl_approved_ids", [])
        state["hitl_rejected_ids"] = state.get("hitl_rejected_ids", [])
        
        if requiring_approval:
            logger.info(
                "HITL gate: %d actions pending approval",
                len(requiring_approval),
            )
            # Trace HITL gate event
            TracingCallbackHandler.on_hitl_gate(
                state.get("thread_id", "unknown"),
                len(requiring_approval),
            )
        else:
            logger.info("HITL gate: no actions pending, continuing")
        
        return state

    def _route_after_hitl(self, state: GraphState) -> str:
        """
        Conditional routing after HITL gate.
        
        Returns:
            - "wait": Pause execution, wait for human approval
            - "execute": Execute approved actions
            - "skip": No actions needed, continue to memory recording
        """
        # Check if this is a resumed execution with approved actions
        approved_ids = state.get("hitl_approved_ids", [])
        if state.get("hitl_resumed") and approved_ids:
            logger.info("Resuming with %d approved actions", len(approved_ids))
            return "execute"
        
        # Check if there are pending actions needing approval
        if state.get("hitl_wait"):
            logger.info("HITL: Pausing for human approval")
            return "wait"
        
        # No pending actions, skip execution
        return "skip"

    async def _execute_approved_node(self, state: GraphState) -> GraphState:
        """
        Execute actions that have been approved by humans.
        
        This node is only reached after resuming from HITL wait.
        """
        approved_ids = state.get("hitl_approved_ids", [])
        
        if not approved_ids:
            logger.info("No approved actions to execute")
            return state
        
        logger.info("Executing %d approved actions: %s", len(approved_ids), approved_ids)
        
        # The actual execution happens via the ActionExecutor in the HITL service
        # Here we just log and update state
        executed = []
        for action_id in approved_ids:
            try:
                # Note: Actual execution is done via /actions/execute endpoint
                # This node tracks what was executed for memory recording
                executed.append(action_id)
                logger.info("Action %d marked as executed", action_id)
            except Exception as exc:
                logger.error("Failed to track action %d: %s", action_id, exc)
                if "system_warnings" not in state:
                    state["system_warnings"] = []
                state["system_warnings"].append(f"Action {action_id} execution tracking failed: {exc}")
        
        # Update state with execution results
        state["hitl_approved_ids"] = []  # Clear after processing
        state["hitl_resumed"] = False
        
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
        thread_id: str | None = None,
    ) -> tuple[SupervisorOutput, str, bool]:
        """
        Run the operations graph for a query.
        
        Args:
            query: The user's question
            conversation_history: Previous conversation messages
            metadata: Additional metadata for the query
            thread_id: Optional thread ID for tracking (auto-generated if not provided)
            
        Returns:
            Tuple of (SupervisorOutput, thread_id, hitl_waiting)
            - hitl_waiting: True if graph paused for human approval
        """
        # Generate or use provided thread_id
        thread_id = thread_id or str(uuid.uuid4())
        
        initial_state = self._supervisor.initialize_state(query, conversation_history)
        initial_state["thread_id"] = thread_id
        initial_state["hitl_resumed"] = False
        
        if metadata:
            initial_state["metadata"] = metadata

        # Run with thread config for checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            # Check if this was an interrupt (HITL wait)
            logger.info("Graph execution paused or completed: %s", type(exc).__name__)
            # Get the current state from checkpoint
            checkpoint = self._checkpointer.get(config)
            if checkpoint:
                final_state = checkpoint.get("channel_values", {})
            else:
                raise

        hitl_waiting = final_state.get("hitl_wait", False)
        
        return (
            SupervisorOutput(
                summary=final_state.get("diagnosis"),
                answer=final_state.get("_final_answer", "No answer generated."),
                diagnostics=final_state.get("_diagnostics", []),
                pending_actions=final_state.get("pending_action_proposals", []),
            ),
            thread_id,
            hitl_waiting,
        )

    async def resume(
        self,
        thread_id: str,
        approved_action_ids: list[int] | None = None,
        rejected_action_ids: list[int] | None = None,
    ) -> SupervisorOutput:
        """
        Resume graph execution after human approval/rejection of actions.
        
        Args:
            thread_id: The thread ID from the original run
            approved_action_ids: List of action IDs that were approved
            rejected_action_ids: List of action IDs that were rejected
            
        Returns:
            SupervisorOutput with final results
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current checkpoint state
        checkpoint = self._checkpointer.get(config)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id: {thread_id}")
        
        current_state = checkpoint.get("channel_values", {})
        
        # Update state with approval decisions
        current_state["hitl_approved_ids"] = approved_action_ids or []
        current_state["hitl_rejected_ids"] = rejected_action_ids or []
        current_state["hitl_resumed"] = True
        current_state["hitl_wait"] = False  # Clear wait flag
        
        logger.info(
            "Resuming thread %s with %d approved, %d rejected actions",
            thread_id,
            len(approved_action_ids or []),
            len(rejected_action_ids or []),
        )

        # Trace HITL resume event
        TracingCallbackHandler.on_hitl_resume(
            thread_id,
            len(approved_action_ids or []),
            len(rejected_action_ids or []),
        )

        # Resume execution from checkpoint
        final_state = await self._graph.ainvoke(current_state, config=config)
        
        return SupervisorOutput(
            summary=final_state.get("diagnosis"),
            answer=final_state.get("_final_answer", "No answer generated."),
            diagnostics=final_state.get("_diagnostics", []),
            pending_actions=final_state.get("pending_action_proposals", []),
        )

    def get_pending_thread_state(self, thread_id: str) -> GraphState | None:
        """
        Get the current state of a paused thread.
        
        Useful for inspecting what actions are pending approval.
        """
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self._checkpointer.get(config)
        
        if not checkpoint:
            return None
        
        return checkpoint.get("channel_values", {})
