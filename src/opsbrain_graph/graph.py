from __future__ import annotations

import asyncio
import json
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
        graph.add_node("reject_off_topic", self._reject_off_topic_node)
        graph.add_node("run_tasks", self._run_tasks_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("replan", self._replan_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("hitl_gate", self._hitl_gate_node)
        graph.add_node("execute_approved", self._execute_approved_node)
        graph.add_node("post_hitl_synthesize", self._post_hitl_synthesize_node)
        graph.add_node("record_memory", self._record_memory_node)

        # Define edges with conditional routing
        graph.add_edge(START, "plan")

        # After planning, check if query is off-topic (empty battle plan)
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "run_tasks": "run_tasks",
                "reject": "reject_off_topic",
            },
        )

        # Off-topic rejection goes straight to end
        graph.add_edge("reject_off_topic", END)

        graph.add_edge("run_tasks", "evaluate")

        # Conditional edge: evaluate results and decide next step
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "synthesize": "synthesize",
                "replan": "replan",
            },
        )

        # After re-planning, run the new tasks
        graph.add_edge("replan", "run_tasks")

        # Synthesis leads to reflection for quality checking
        graph.add_edge("synthesize", "reflect")

        # Reflection leads to HITL gate
        graph.add_edge("reflect", "hitl_gate")

        # Conditional edge: HITL gate decides if we wait or continue
        graph.add_conditional_edges(
            "hitl_gate",
            self._route_after_hitl,
            {
                "execute": "execute_approved",  # Actions approved, execute them
                "skip": "record_memory",  # No actions needed
            },
        )

        # After executing approved actions, re-synthesize with results
        graph.add_edge("execute_approved", "post_hitl_synthesize")

        # After post-HITL synthesis, record memory
        graph.add_edge("post_hitl_synthesize", "record_memory")

        # After recording memory, end
        graph.add_edge("record_memory", END)

        # Compile with checkpointer for HITL state persistence
        # Interrupt BEFORE execute_approved - this is where we wait for human approval
        return graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["execute_approved"],
        )

    async def _plan_node(self, state: GraphState) -> GraphState:
        logger.info("Planning tasks for query: %s", state["user_query"])
        tasks = await self._supervisor.plan(state)  # Now async with LLM-based planning
        state["battle_plan"] = tasks
        logger.info("Battle plan: %s", [t.agent for t in tasks])
        return state

    def _route_after_plan(self, state: GraphState) -> str:
        """Route after planning: reject off-topic queries or proceed with tasks."""
        battle_plan = state.get("battle_plan", [])
        if not battle_plan:
            logger.info("Empty battle plan - query appears to be off-topic")
            return "reject"
        return "run_tasks"

    async def _reject_off_topic_node(self, state: GraphState) -> GraphState:
        """Handle off-topic queries with a polite rejection message."""
        logger.info("Rejecting off-topic query: %s", state["user_query"])

        rejection_message = (
            "I'm sorry, but I can only help with e-commerce operations questions. "
            "This includes:\n\n"
            "- **Sales & Revenue**: Sales trends, top products, order analysis\n"
            "- **Inventory**: Stock levels, low-stock alerts, product availability\n"
            "- **Marketing**: Campaign performance, ad spend, ROAS\n"
            "- **Customer Support**: Ticket analysis, sentiment, issue trends\n\n"
            "Please ask a question related to your business operations, and I'll be happy to help!"
        )

        state["_final_answer"] = rejection_message
        state["_diagnostics"] = ["Query rejected: off-topic"]
        state["diagnosis"] = None
        state["pending_action_proposals"] = []

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
        logger.info(
            "Executing %d agents in parallel: %s",
            len(valid_tasks),
            [t[0].agent for t in valid_tasks],
        )

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

    async def _reflect_node(self, state: GraphState) -> GraphState:
        """
        Reflection node: evaluate the quality and completeness of the synthesis.

        This detects missing information, weak conclusions, and adjusts confidence.
        """
        logger.info("Reflecting on synthesis quality...")

        reflection = await self._supervisor.reflect(state)
        self._supervisor.apply_reflection(state, reflection)

        is_complete = reflection.get("is_complete", True)
        confidence_adj = reflection.get("confidence_adjustment", 0)

        logger.info(
            "Reflection complete: is_complete=%s, confidence_adjustment=%.2f, reasoning=%s",
            is_complete,
            confidence_adj,
            reflection.get("reasoning", "N/A")[:100],
        )

        # Add reflection info to diagnostics
        if "_diagnostics" not in state:
            state["_diagnostics"] = []

        if not is_complete:
            missing = reflection.get("missing_information", [])
            if missing:
                state["_diagnostics"].append(
                    f"Reflection detected missing info: {', '.join(missing[:3])}"
                )

        if confidence_adj != 0:
            state["_diagnostics"].append(
                f"Confidence adjusted by {confidence_adj:+.2f} after reflection"
            )

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
            - "execute": Has pending/approved actions, go to execute_approved
                        (will interrupt there if actions pending approval)
            - "skip": No actions at all, skip to memory recording
        """
        proposals = state.get("pending_action_proposals", [])
        approved_ids = state.get("hitl_approved_ids", [])

        # If there are approved actions (resumed flow) or pending proposals
        if approved_ids or proposals:
            logger.info(
                "HITL routing to execute: approved=%d, proposals=%d",
                len(approved_ids),
                len(proposals),
            )
            return "execute"

        # No actions at all, skip execution
        logger.info("HITL: No actions, skipping to memory recording")
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
                state["system_warnings"].append(
                    f"Action {action_id} execution tracking failed: {exc}"
                )

        # Store executed actions for memory recording
        state["_executed_action_ids"] = executed

        # Update state with execution results
        state["hitl_approved_ids"] = []  # Clear after processing
        state["hitl_resumed"] = False

        return state

    async def _post_hitl_synthesize_node(self, state: GraphState) -> GraphState:
        """
        Re-synthesize the answer after HITL actions have been executed.

        This node takes the execution results from hitl_execution_results
        and generates a comprehensive answer incorporating the new data.
        """
        execution_results = state.get("hitl_execution_results", [])
        original_answer = state.get("_final_answer", "")
        user_query = state.get("user_query", "")

        if not execution_results:
            logger.info("No HITL execution results to synthesize")
            return state

        logger.info(
            "Post-HITL synthesis: incorporating %d execution result(s)",
            len(execution_results),
        )

        # Build context for LLM re-synthesis
        context_parts = []
        context_parts.append(f"ORIGINAL USER QUESTION: {user_query}\n")
        context_parts.append(f"INITIAL ANALYSIS (before data retrieval):\n{original_answer}\n")
        context_parts.append("=" * 50)
        context_parts.append("\nEXECUTED ACTION RESULTS:")

        for i, result in enumerate(execution_results, 1):
            action_type = result.get("action_type", "unknown")
            success = result.get("success", False)
            result_data = result.get("result", {})

            context_parts.append(f"\n--- Action {i}: {action_type} ---")
            context_parts.append(f"Success: {success}")

            if success and result_data:
                # Extract meaningful data from the result
                if "result" in result_data:
                    inner_result = result_data["result"]
                    if "rows" in inner_result:
                        rows = inner_result["rows"]
                        context_parts.append(f"Data returned: {len(rows)} row(s)")
                        # Include the actual data for analysis
                        context_parts.append(f"Data:\n{json.dumps(rows, indent=2, default=str)}")
                    else:
                        context_parts.append(
                            f"Result: {json.dumps(inner_result, indent=2, default=str)}"
                        )
                else:
                    context_parts.append(
                        f"Result: {json.dumps(result_data, indent=2, default=str)}"
                    )
            elif not success:
                error_msg = result.get("message", "Unknown error")
                context_parts.append(f"Error: {error_msg}")

        context = "\n".join(context_parts)

        # Generate new synthesized answer
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            # Use a post-HITL synthesis prompt
            system_prompt = """You are an AI assistant that analyzes e-commerce operations data.

The user asked a question, and we initially provided a preliminary analysis. Now we have 
executed the data retrieval actions (SQL queries, API calls, etc.) and have the actual data.

Your task is to provide a COMPREHENSIVE FINAL ANSWER that:
1. Directly answers the user's original question using the actual data
2. Provides specific numbers, insights, and analysis from the execution results
3. Draws meaningful conclusions and patterns from the data
4. If the data reveals anything significant or actionable, highlight it
5. Be specific with numbers - don't just say "several" when you have exact counts

Format your response in clear, readable markdown with:
- A direct answer to the question upfront
- Supporting data and analysis
- Key insights or recommendations if relevant

Do NOT reference "execution results" or technical details - just present the analysis naturally."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context),
            ]

            response = await self._supervisor.llm.ainvoke(messages)
            new_answer = response.content

            # Update state with the new synthesized answer
            state["_final_answer"] = new_answer

            # Update diagnostics
            if "_diagnostics" not in state:
                state["_diagnostics"] = []
            state["_diagnostics"].append(
                f"Post-HITL re-synthesis: analyzed {len(execution_results)} execution result(s)"
            )

            logger.info("Post-HITL synthesis complete, answer updated")

        except Exception as exc:
            logger.exception("Post-HITL synthesis failed: %s", exc)
            # On failure, keep the original answer but append a note about the data
            state["_final_answer"] = (
                f"{original_answer}\n\n---\n\n"
                f"**Note:** The requested actions were executed successfully. "
                f"Retrieved {len(execution_results)} result(s)."
            )

        return state

    async def _record_memory_node(self, state: GraphState) -> GraphState:
        """
        Record the incident to memory for future reference.

        This enables learning from past incidents and answering
        "What did we do last time?" style queries.

        Stores structured action history including:
        - Actions proposed by agents
        - Actions approved by user (HITL)
        - Actions rejected by user
        - Outcome and confidence
        """
        logger.info("=== RECORD MEMORY NODE ENTERED ===")
        diagnosis = state.get("diagnosis")

        # Log diagnosis state for debugging
        if diagnosis:
            logger.info(
                "Memory recording check: confidence=%.2f (threshold=0.5)", diagnosis.confidence
            )
        else:
            logger.warning("Memory recording skipped: no diagnosis available")
            return state

        # Lower threshold to 0.5 to capture more incidents
        if diagnosis.confidence >= 0.5:
            try:
                # Build action_taken (text summary for backward compatibility)
                action_taken = self._build_action_summary(state)

                # Build outcome from execution results
                outcome = self._build_outcome_summary(state)

                # Build structured action data for "what did we do" queries
                actions_proposed = self._extract_proposed_actions(state)
                actions_approved = self._extract_approved_actions(state)
                actions_rejected = self._extract_rejected_actions(state)

                incident = MemoryIncident(
                    incident_summary=state["user_query"],
                    root_cause=diagnosis.narrative[:500] if diagnosis.narrative else None,
                    action_taken=action_taken,
                    outcome=outcome,
                    actions_proposed=actions_proposed,
                    actions_approved=actions_approved,
                    actions_rejected=actions_rejected,
                    confidence_score=diagnosis.confidence,
                )
                logger.info(
                    "Saving memory incident: query='%s', confidence=%.2f",
                    state["user_query"][:50],
                    diagnosis.confidence,
                )
                memory_id = await self._memory_service.save_incident(incident)
                logger.info(
                    "Recorded memory with ID: %s (proposed=%d, approved=%d, rejected=%d)",
                    memory_id,
                    len(actions_proposed),
                    len(actions_approved),
                    len(actions_rejected),
                )
            except Exception as exc:
                logger.exception("Failed to record memory: %s", exc)
        else:
            logger.info(
                "Memory recording skipped: confidence %.2f below threshold 0.5",
                diagnosis.confidence,
            )
        return state

    def _extract_proposed_actions(self, state: GraphState) -> list[dict]:
        """Extract structured list of proposed actions from recommendations."""
        recommendations = state.get("recommendations", [])
        actions = []
        for rec in recommendations:
            actions.append(
                {
                    "action_type": rec.action_type,
                    "reasoning": rec.reasoning[:200] if rec.reasoning else None,
                    "agent": rec.agent_name if hasattr(rec, "agent_name") else None,
                    "requires_approval": rec.requires_approval,
                }
            )
        return actions

    def _extract_approved_actions(self, state: GraphState) -> list[str]:
        """Extract list of action types that were approved."""
        # Get approved action IDs
        approved_ids = state.get("hitl_approved_ids", [])
        if not approved_ids:
            return []

        # Map back to action types from pending_action_proposals
        pending = state.get("pending_action_proposals", [])
        approved_types = []
        for action in pending:
            action_id = (
                action.get("id") if isinstance(action, dict) else getattr(action, "id", None)
            )
            if action_id in approved_ids:
                action_type = (
                    action.get("action_type")
                    if isinstance(action, dict)
                    else getattr(action, "action_type", None)
                )
                if action_type:
                    approved_types.append(action_type)

        return approved_types

    def _extract_rejected_actions(self, state: GraphState) -> list[str]:
        """Extract list of action types that were rejected."""
        # Get rejected action IDs
        rejected_ids = state.get("hitl_rejected_ids", [])
        if not rejected_ids:
            return []

        # Map back to action types from pending_action_proposals
        pending = state.get("pending_action_proposals", [])
        rejected_types = []
        for action in pending:
            action_id = (
                action.get("id") if isinstance(action, dict) else getattr(action, "id", None)
            )
            if action_id in rejected_ids:
                action_type = (
                    action.get("action_type")
                    if isinstance(action, dict)
                    else getattr(action, "action_type", None)
                )
                if action_type:
                    rejected_types.append(action_type)

        return rejected_types

    def _build_action_summary(self, state: GraphState) -> str | None:
        """Build a summary of actions taken/recommended from the state."""
        parts = []

        # Include recommendations made by agents
        recommendations = state.get("recommendations", [])
        if recommendations:
            rec_summaries = []
            for rec in recommendations[:5]:  # Limit to 5
                rec_summaries.append(f"{rec.action_type}: {rec.reasoning[:100]}")
            parts.append("Recommendations: " + "; ".join(rec_summaries))

        # Include approved/executed actions
        executed = state.get("_executed_action_ids", [])
        if executed:
            parts.append(f"Executed action IDs: {executed}")

        # Include rejected actions for context
        rejected = state.get("hitl_rejected_ids", [])
        if rejected:
            parts.append(f"Rejected action IDs: {rejected}")

        return " | ".join(parts) if parts else None

    def _build_outcome_summary(self, state: GraphState) -> str | None:
        """Build a summary of the outcome for memory recording."""
        parts = []

        # Reflection results indicate quality
        reflection = state.get("_reflection")
        if reflection:
            if reflection.get("is_complete"):
                parts.append("Analysis was complete")
            else:
                missing = reflection.get("missing_information", [])
                if missing:
                    parts.append(f"Incomplete - missing: {', '.join(missing[:3])}")

        # Any warnings encountered
        warnings = state.get("system_warnings", [])
        if warnings:
            # Filter out reflection warnings (already captured above)
            non_reflection_warnings = [w for w in warnings if not w.startswith("[Reflection]")]
            if non_reflection_warnings:
                parts.append(f"Warnings: {len(non_reflection_warnings)}")

        # Confidence level gives indication of outcome quality
        diagnosis = state.get("diagnosis")
        if diagnosis:
            if diagnosis.confidence >= 0.85:
                parts.append("High confidence resolution")
            elif diagnosis.confidence >= 0.65:
                parts.append("Medium confidence resolution")
            else:
                parts.append("Low confidence - may need follow-up")

        return " | ".join(parts) if parts else "Completed"

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

        final_state = await self._graph.ainvoke(initial_state, config=config)

        # Check if the graph was interrupted (HITL waiting)
        # When interrupted before execute_approved, the graph state will have
        # pending_action_proposals but won't have reached the end
        snapshot = self._graph.get_state(config)
        hitl_waiting = bool(snapshot.next)  # If there's a next node, we're interrupted

        if hitl_waiting:
            logger.info(
                "Graph interrupted for HITL approval. Next nodes: %s",
                snapshot.next,
            )

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
        execution_results: list[dict[str, Any]] | None = None,
    ) -> SupervisorOutput:
        """
        Resume graph execution after human approval/rejection of actions.

        Args:
            thread_id: The thread ID from the original run
            approved_action_ids: List of action IDs that were approved
            rejected_action_ids: List of action IDs that were rejected
            execution_results: Results from executed HITL actions for re-synthesis

        Returns:
            SupervisorOutput with final results
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get current checkpoint state to verify it exists
        checkpoint = self._checkpointer.get(config)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id: {thread_id}")

        logger.info(
            "Resuming thread %s with %d approved, %d rejected actions",
            thread_id,
            len(approved_action_ids or []),
            len(rejected_action_ids or []),
        )

        # Store execution results for post-HITL synthesis
        if execution_results:
            logger.info(
                "Storing %d execution result(s) for post-HITL synthesis",
                len(execution_results),
            )

        # Trace HITL resume event
        TracingCallbackHandler.on_hitl_resume(
            thread_id,
            len(approved_action_ids or []),
            len(rejected_action_ids or []),
        )

        # Use update_state to modify the checkpoint state at the interrupted node
        # This is the correct way to update state before resuming in LangGraph
        state_updates = {
            "hitl_approved_ids": approved_action_ids or [],
            "hitl_rejected_ids": rejected_action_ids or [],
            "hitl_resumed": True,
            "hitl_wait": False,  # Clear wait flag
        }

        if execution_results:
            state_updates["hitl_execution_results"] = execution_results

        # Update state at the checkpoint (before execute_approved node)
        self._graph.update_state(config, state_updates)

        # Resume execution from checkpoint by passing None
        # This tells LangGraph to continue from where it was interrupted
        final_state = await self._graph.ainvoke(None, config=config)

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
