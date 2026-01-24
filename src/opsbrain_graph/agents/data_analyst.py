from __future__ import annotations

from opsbrain_graph.tools import ExecuteSQLRequest
from .base_agent import AgentRunContext, AgentTask, BaseAgent


class DataAnalystAgent(BaseAgent):
    name = "data_analyst"
    description = "Executes complex SQL tasks for cross-domain insights."

    async def run(self, task: AgentTask, context: AgentRunContext):
        statement = task.parameters.get("statement")
        params = task.parameters.get("params")

        if not statement:
            return self.failure("Data Analyst task requires 'statement' parameter.")

        try:
            result = await self.tools.sql.execute(
                ExecuteSQLRequest(statement=statement, params=params)
            )
        except Exception as exc:
            return self.failure(exc)

        findings = {"columns": result.columns, "rows": result.rows}
        insights = [f"Returned {len(result.rows)} rows."]
        return self.success(findings=findings, insights=insights)
