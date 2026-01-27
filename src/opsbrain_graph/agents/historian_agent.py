from __future__ import annotations

from opsbrain_graph.memory import MemoryIncident
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)


class HistorianAgent(BaseAgent):
    name = "historian"
    description = "Retrieves and stores lessons learned."

    metadata = AgentMetadata(
        name="historian",
        display_name="HISTORIAN",
        description="Retrieves similar past incidents from memory for context. Stores new incidents as lessons learned. Essential for 'why' questions.",
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
        ],
        priority_boost=["root cause", "explain why"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext):
        if not self.memory_service:
            return self.failure("Memory service unavailable.")

        mode = task.parameters.get("mode", "query")

        if mode == "query":
            query = task.parameters.get("query") or context.user_query
            k = task.parameters.get("k", 3)
            hits = await self.memory_service.query_similar_incidents(query, k)
            matches = [hit.to_dict() for hit in hits]
            insights = (
                [f"Historian: retrieved {len(matches)} similar incidents."]
                if matches
                else ["Historian: no close incidents found."]
            )
            return self.success(findings={"matches": matches}, insights=insights)

        if mode == "save":
            incident_payload = task.parameters.get("incident")
            if not incident_payload:
                return self.failure("Historian save mode requires 'incident' payload.")
            incident = MemoryIncident(**incident_payload)
            memory_id = await self.memory_service.save_incident(incident)
            return self.success(
                findings={"memory_id": memory_id},
                insights=[f"Incident persisted with id={memory_id}."],
            )

        return self.failure(f"Unknown historian mode '{mode}'.")
