from __future__ import annotations

from opsbrain_graph.tools import ExecuteSQLRequest
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)


class DataAnalystAgent(BaseAgent):
    name = "data_analyst"
    description = "Executes complex SQL tasks for cross-domain insights."

    metadata = AgentMetadata(
        name="data_analyst",
        display_name="DATA_ANALYST",
        description="Performs custom SQL queries for complex analysis that doesn't fit other specialized agents. Use for cross-domain insights or custom reports.",
        capabilities=[
            AgentCapability(
                name="custom_query",
                description="Execute arbitrary analytical SQL queries",
                parameters={
                    "statement": "SQL SELECT statement to execute (required)",
                    "params": "Optional query parameters dict",
                },
                example_queries=[
                    "Run a custom analysis on orders by region",
                    "Show me a breakdown of revenue by channel",
                    "Complex query: compare this month to last month",
                ],
            ),
        ],
        keywords=["query", "sql", "analyze", "report", "breakdown", "custom", "compare"],
        priority_boost=["complex analysis", "custom report"],
    )

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
