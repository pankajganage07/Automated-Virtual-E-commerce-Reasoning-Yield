from __future__ import annotations

from opsbrain_graph.memory import MemoryIncident
from .base_agent import AgentRunContext, AgentTask, BaseAgent


class HistorianAgent(BaseAgent):
    name = "historian"
    description = "Retrieves and stores lessons learned."

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
