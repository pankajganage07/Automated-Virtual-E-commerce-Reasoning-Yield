from __future__ import annotations

from typing import Any

from opsbrain_graph.tools import ExecuteSQLRequest
from .base_agent import AgentRunContext, AgentTask, BaseAgent


class SalesAgent(BaseAgent):
    name = "sales"
    description = "Analyzes revenue, trends, and anomalies."

    async def run(self, task: AgentTask, context: AgentRunContext):
        params = task.parameters
        window_days = params.get("window_days", 7)
        group_by = params.get("group_by", "day")

        statement = """
        SELECT date_trunc(:group_by, timestamp) AS bucket,
               SUM(revenue) AS revenue,
               SUM(qty) AS units
        FROM orders
        WHERE timestamp >= NOW() - make_interval(days => :window_days)
        GROUP BY bucket
        ORDER BY bucket DESC;
        """

        try:
            resp = await self.tools.sql.execute(
                ExecuteSQLRequest(
                    statement=statement,
                    params={"group_by": group_by, "window_days": window_days},
                )
            )
        except Exception as exc:
            return self.failure(exc)

        rows = resp.rows

        insights = []
        if rows:
            latest = rows[0]["revenue"]
            prev = rows[1]["revenue"] if len(rows) > 1 else None
            if prev:
                delta = ((latest - prev) / prev) * 100 if prev else 0
                insights.append(
                    f"Latest {group_by} revenue {latest:.2f}, change {delta:+.1f}% vs prior."
                )

        findings = {"trend": rows}
        return self.success(findings=findings, insights=insights)
