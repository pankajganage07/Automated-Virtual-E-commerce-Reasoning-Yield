from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.memory import MemoryIncident, MemoryHitWithActions
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
                summary = match.get("incident_summary", "No summary")[:100]
                score = match.get("score", 0)
                insights.append(f"  {i}. (Similarity: {score:.0%}) {summary}...")

                # Include root cause if available
                root_cause = match.get("root_cause")
                if root_cause:
                    insights.append(f"     Root cause: {root_cause[:80]}...")

                # Show if actions were taken
                if match.get("actions_approved"):
                    insights.append(
                        f"     Actions approved: {', '.join(match['actions_approved'][:3])}"
                    )
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
            # Use the new action-focused query
            hits = await self.memory_service.query_incidents_with_actions(
                query, k=k, only_with_actions=True
            )
        except Exception as exc:
            logger.exception("historian agent (past_actions) failed: %s", exc)
            return self.failure(exc)

        # Format the results with action history
        past_incidents: list[dict[str, Any]] = []
        for hit in hits:
            incident_info = {
                "incident_summary": hit.incident_summary[:200],
                "root_cause": hit.root_cause[:150] if hit.root_cause else None,
                "similarity": hit.similarity_score,
                "when": hit.created_at.strftime("%Y-%m-%d") if hit.created_at else "Unknown",
                "actions_proposed": hit.action_history.proposed,
                "actions_approved": hit.action_history.approved,
                "actions_rejected": hit.action_history.rejected,
                "action_summary": hit.action_history.summary,
                "outcome": hit.outcome,
                "confidence": hit.confidence_score,
            }
            past_incidents.append(incident_info)

        findings: dict[str, Any] = {
            "query": query,
            "total_matches": len(hits),
            "past_incidents": past_incidents,
        }
        insights: list[str] = []

        if past_incidents:
            insights.append(f"📜 Found {len(past_incidents)} similar past incidents with actions:")
            for i, incident in enumerate(past_incidents, 1):
                insights.append(
                    f"\n**{i}. {incident['when']}** (Similarity: {incident['similarity']:.0%})"
                )
                insights.append(f"   Situation: {incident['incident_summary']}...")

                if incident["root_cause"]:
                    insights.append(f"   Root cause: {incident['root_cause']}...")

                # Show action history
                if incident["actions_approved"]:
                    insights.append(
                        f"   ✅ Approved actions: {', '.join(incident['actions_approved'])}"
                    )
                if incident["actions_rejected"]:
                    insights.append(
                        f"   ❌ Rejected actions: {', '.join(incident['actions_rejected'])}"
                    )
                if incident["action_summary"] and not incident["actions_approved"]:
                    insights.append(f"   Actions taken: {incident['action_summary'][:100]}...")

                if incident["outcome"]:
                    insights.append(f"   Outcome: {incident['outcome']}")

            # Analyze what worked
            all_approved = []
            all_rejected = []
            for inc in past_incidents:
                all_approved.extend(inc.get("actions_approved") or [])
                all_rejected.extend(inc.get("actions_rejected") or [])

            if all_approved:
                # Count occurrences
                from collections import Counter

                approved_counts = Counter(all_approved)
                top_actions = approved_counts.most_common(3)
                if top_actions:
                    insights.append(
                        f"\n📊 **Most commonly approved actions:** {', '.join(f'{a}({c}x)' for a,c in top_actions)}"
                    )
        else:
            insights.append("📜 No similar incidents with action history found.")
            insights.append("   This may be the first time this type of situation has occurred.")

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
