from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.memory import MemoryIncident
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentResult,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)

logger = logging.getLogger("agent.historian")


class HistorianAgent(BaseAgent):
    name = "historian"
    description = "Retrieves and stores lessons learned."

    metadata = AgentMetadata(
        name="historian",
        display_name="HISTORIAN",
        description="Retrieves similar past incidents from memory for context. Stores new incidents as lessons learned. Can find what actions worked before.",
        capabilities=[
            AgentCapability(
                name="query",
                description="Search for similar past incidents using semantic similarity",
                parameters={
                    "query": "Search query (defaults to user's question)",
                    "k": "Number of results to return (default: 3)",
                },
                example_queries=[
                    "Has this happened before?",
                    "Why did sales drop last time?",
                    "What caused similar issues in the past?",
                ],
            ),
            AgentCapability(
                name="past_actions",
                description="Find what actions were taken in similar past incidents and their outcomes",
                parameters={
                    "query": "Search query for finding relevant past incidents",
                    "k": "Number of past incidents to search (default: 5)",
                },
                example_queries=[
                    "What did we do last time sales dropped?",
                    "Did discounts help previously?",
                    "Which actions worked best in past incidents?",
                ],
            ),
            AgentCapability(
                name="save",
                description="Store a new incident as a lesson learned",
                parameters={
                    "incident": "Incident details (summary, root_cause, action_taken, outcome)",
                },
                example_queries=[
                    "Remember this incident for future reference",
                    "Save this as a lesson learned",
                ],
            ),
        ],
        keywords=[
            "why",
            "reason",
            "cause",
            "explain",
            "happened",
            "history",
            "before",
            "similar",
            "past",
            "last time",
            "previously",
            "worked",
        ],
        priority_boost=["root cause", "explain why", "what did we do"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        if not self.memory_service:
            return self.failure("Memory service unavailable.")

        mode = task.parameters.get("mode", "query")

        if mode == "query":
            return await self._run_query(task.parameters, context)
        elif mode == "past_actions":
            return await self._run_past_actions(task.parameters, context)
        elif mode == "save":
            return await self._run_save(task.parameters)
        else:
            return self.failure(f"Unknown historian mode '{mode}'.")

    async def _run_query(self, params: dict[str, Any], context: AgentRunContext) -> AgentResult:
        """Search for similar past incidents."""
        query = params.get("query") or context.user_query
        k = params.get("k", 3)

        try:
            hits = await self.memory_service.query_similar_incidents(query, k)
        except Exception as exc:
            logger.exception("historian agent (query) failed: %s", exc)
            return self.failure(exc)

        matches = [hit.to_dict() for hit in hits]
        findings: dict[str, Any] = {"matches": matches, "query": query}
        insights: list[str] = []

        if matches:
            insights.append(f"📚 Found {len(matches)} similar past incidents:")
            for i, match in enumerate(matches, 1):
                summary = match.get("summary", "No summary")[:100]
                similarity = match.get("similarity", 0)
                insights.append(f"  {i}. (Similarity: {similarity:.0%}) {summary}...")

                # Include root cause if available
                root_cause = match.get("root_cause")
                if root_cause:
                    insights.append(f"     Root cause: {root_cause[:80]}...")
        else:
            insights.append("📚 No similar incidents found in memory.")

        return self.success(findings=findings, insights=insights)

    async def _run_past_actions(
        self, params: dict[str, Any], context: AgentRunContext
    ) -> AgentResult:
        """Find what actions were taken in similar past incidents."""
        query = params.get("query") or context.user_query
        k = params.get("k", 5)

        try:
            hits = await self.memory_service.query_similar_incidents(query, k)
        except Exception as exc:
            logger.exception("historian agent (past_actions) failed: %s", exc)
            return self.failure(exc)

        matches = [hit.to_dict() for hit in hits]

        # Extract actions and outcomes
        actions_taken: list[dict[str, Any]] = []
        for match in matches:
            action = match.get("action_taken")
            outcome = match.get("outcome")
            if action:
                actions_taken.append(
                    {
                        "incident_summary": match.get("summary", "Unknown")[:100],
                        "action_taken": action,
                        "outcome": outcome or "Unknown",
                        "similarity": match.get("similarity", 0),
                    }
                )

        findings: dict[str, Any] = {
            "query": query,
            "total_matches": len(matches),
            "actions_found": len(actions_taken),
            "past_actions": actions_taken,
        }
        insights: list[str] = []

        if actions_taken:
            insights.append(f"📜 Found {len(actions_taken)} past actions from similar incidents:")
            for i, action_info in enumerate(actions_taken, 1):
                insights.append(f"  {i}. Incident: {action_info['incident_summary']}...")
                insights.append(f"     Action: {action_info['action_taken']}")
                insights.append(f"     Outcome: {action_info['outcome']}")

            # Analyze which actions had positive outcomes
            positive_outcomes = [
                a
                for a in actions_taken
                if "success" in str(a.get("outcome", "")).lower()
                or "resolved" in str(a.get("outcome", "")).lower()
                or "improved" in str(a.get("outcome", "")).lower()
            ]
            if positive_outcomes:
                insights.append(f"✅ {len(positive_outcomes)} actions had positive outcomes")
        else:
            insights.append("📜 No past actions found for similar incidents.")

        return self.success(findings=findings, insights=insights)

    async def _run_save(self, params: dict[str, Any]) -> AgentResult:
        """Store a new incident as a lesson learned."""
        incident_payload = params.get("incident")
        if not incident_payload:
            return self.failure("Historian save mode requires 'incident' payload.")

        try:
            incident = MemoryIncident(**incident_payload)
            memory_id = await self.memory_service.save_incident(incident)
        except Exception as exc:
            logger.exception("historian agent (save) failed: %s", exc)
            return self.failure(exc)

        return self.success(
            findings={"memory_id": memory_id},
            insights=[f"✅ Incident persisted with id={memory_id}."],
        )
