from __future__ import annotations

from langgraph.tools import QueryVectorMemoryRequest, SaveMemoryRequest
from .base_agent import AgentRunContext, AgentTask, BaseAgent


class HistorianAgent(BaseAgent):
    name = "historian"
    description = "Retrieves and stores lessons learned."

    async def run(self, task: AgentTask, context: AgentRunContext):
        mode = task.parameters.get("mode", "query")
        if mode == "query":
            query = task.parameters.get("query")
            if not query:
                return self.failure("Historian query mode requires 'query'.")
            try:
                resp = await self.tools.memory.query_memory(QueryVectorMemoryRequest(query=query))
            except Exception as exc:
                return self.failure(exc)
            findings = {"matches": [hit.model_dump() for hit in resp.results]}
            insights = [f"{len(resp.results)} analogous incidents retrieved."]
            return self.success(findings=findings, insights=insights)

        if mode == "save":
            incident = task.parameters.get("incident")
            if not incident:
                return self.failure("Historian save mode requires 'incident'.")
            try:
                resp = await self.tools.memory.save_memory(SaveMemoryRequest(**incident))
            except Exception as exc:
                return self.failure(exc)
            findings = {"memory_id": resp.memory_id, "message": resp.message}
            return self.success(findings=findings, insights=["Incident saved to memory store."])

        return self.failure(f"Unknown historian mode '{mode}'.")
