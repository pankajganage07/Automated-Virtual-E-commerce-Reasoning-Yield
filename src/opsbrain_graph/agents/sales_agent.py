from __future__ import annotations

import logging
from typing import Any

from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentResult,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)
from opsbrain_graph.tools.sql_tools import (
    ExecuteSQLRequest,
    GetTopProductsRequest,
    GetSalesSummaryRequest,
)

logger = logging.getLogger("agent.sales")


class SalesAgent(BaseAgent):
    name = "sales"
    description = "Analyzes revenue, trends, top products, and anomalies."

    metadata = AgentMetadata(
        name="sales",
        display_name="SALES",
        description="Analyzes revenue trends, sales performance, and detects anomalies. Can identify top-selling products and revenue patterns.",
        capabilities=[
            AgentCapability(
                name="trends",
                description="Analyze revenue trends over time, detect drops or spikes",
                parameters={
                    "window_days": "Number of days to analyze (default: 7)",
                    "group_by": "Grouping: 'day', 'week', or 'month' (default: 'day')",
                },
                example_queries=[
                    "How are sales trending this week?",
                    "Show me revenue trends for the last 30 days",
                    "Why did sales drop yesterday?",
                ],
            ),
            AgentCapability(
                name="top_products",
                description="Find best-selling products by revenue or quantity",
                parameters={
                    "window_days": "Time period to analyze (default: 7)",
                    "limit": "Number of products to return (default: 5)",
                },
                example_queries=[
                    "What are the top 5 selling products?",
                    "Which products made the most money this month?",
                    "Best sellers last week",
                ],
            ),
        ],
        keywords=[
            "sale",
            "revenue",
            "drop",
            "trend",
            "income",
            "earning",
            "money",
            "top",
            "best",
            "product",
        ],
        priority_boost=["revenue", "sales drop", "urgent"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "trends")

        if mode == "top_products":
            return await self._run_top_products(params)
        else:
            return await self._run_trends(params)

    async def _run_top_products(self, params: dict[str, Any]) -> AgentResult:
        """Find top selling products."""
        window_days = params.get("window_days", 7)
        limit = params.get("limit", 5)

        try:
            resp = await self.tools.sales.get_top_products(
                GetTopProductsRequest(window_days=window_days, limit=limit)
            )
        except Exception as exc:
            logger.exception("sales agent (top_products) failed: %s", exc)
            return self.failure(exc)

        products = resp.products
        findings: dict[str, Any] = {
            "top_products": [p.model_dump() for p in products],
            "window_days": window_days,
            "limit": limit,
        }
        insights: list[str] = []

        if products:
            insights.append(f"Top {len(products)} selling products in the last {window_days} days:")
            for i, product in enumerate(products, 1):
                insights.append(
                    f"  {i}. {product.name} - ${product.revenue:,.2f} revenue, "
                    f"{product.units_sold} units sold"
                )

            # Calculate total revenue from top products
            total_revenue = sum(p.revenue for p in products)
            findings["total_top_products_revenue"] = total_revenue
            insights.append(
                f"Total revenue from top {len(products)} products: ${total_revenue:,.2f}"
            )
        else:
            insights.append(f"No product sales data found for the last {window_days} days.")

        return self.success(findings=findings, insights=insights)

    async def _run_trends(self, params: dict[str, Any]) -> AgentResult:
        """Analyze revenue trends."""
        window_days = params.get("window_days", 7)
        group_by = params.get("group_by", "day")

        statement = """
        SELECT date_trunc(:group_by, timestamp) AS bucket,
               SUM(revenue) AS revenue,
               SUM(qty) AS units,
               COUNT(*) AS order_count
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
            logger.exception("sales agent failed: %s", exc)
            return self.failure(exc)

        rows = resp.rows
        findings: dict[str, Any] = {"sales_by_period": rows, "window_days": window_days}
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        if rows:
            # Analyze trends
            latest = rows[0] if rows else None
            previous = rows[1] if len(rows) > 1 else None

            if latest and previous:
                latest_revenue = float(latest.get("revenue", 0))
                prev_revenue = float(previous.get("revenue", 0))
                latest_units = int(latest.get("units", 0))
                prev_units = int(previous.get("units", 0))
                latest_orders = int(latest.get("order_count", 0))
                prev_orders = int(previous.get("order_count", 0))

                # Calculate changes
                if prev_revenue > 0:
                    revenue_change_pct = ((latest_revenue - prev_revenue) / prev_revenue) * 100
                else:
                    revenue_change_pct = 0

                if prev_units > 0:
                    units_change_pct = ((latest_units - prev_units) / prev_units) * 100
                else:
                    units_change_pct = 0

                if prev_orders > 0:
                    orders_change_pct = ((latest_orders - prev_orders) / prev_orders) * 100
                else:
                    orders_change_pct = 0

                # Store analysis in findings
                findings["analysis"] = {
                    "latest_period": str(latest.get("bucket")),
                    "previous_period": str(previous.get("bucket")),
                    "latest_revenue": latest_revenue,
                    "previous_revenue": prev_revenue,
                    "revenue_change_pct": round(revenue_change_pct, 2),
                    "latest_units": latest_units,
                    "previous_units": prev_units,
                    "units_change_pct": round(units_change_pct, 2),
                    "latest_orders": latest_orders,
                    "previous_orders": prev_orders,
                    "orders_change_pct": round(orders_change_pct, 2),
                }

                # Generate insights based on analysis
                insights.append(
                    f"Revenue for {latest.get('bucket')}: ${latest_revenue:,.2f} "
                    f"({revenue_change_pct:+.1f}% vs previous period)"
                )
                insights.append(
                    f"Units sold: {latest_units} ({units_change_pct:+.1f}% vs previous period)"
                )
                insights.append(
                    f"Order count: {latest_orders} ({orders_change_pct:+.1f}% vs previous period)"
                )

                # Detect significant drops and provide potential reasons
                if revenue_change_pct < -10:
                    findings["anomaly_detected"] = True
                    findings["anomaly_type"] = "revenue_drop"
                    findings["severity"] = "high" if revenue_change_pct < -25 else "medium"

                    # Analyze potential causes
                    potential_causes = []

                    if units_change_pct < revenue_change_pct:
                        potential_causes.append(
                            "Order volume dropped more than revenue, indicating fewer customers"
                        )
                    elif units_change_pct > revenue_change_pct:
                        potential_causes.append(
                            "Revenue dropped more than volume, suggesting lower average order value or discounting"
                        )

                    if orders_change_pct < -10:
                        potential_causes.append(
                            f"Order count decreased by {orders_change_pct:.1f}%, indicating reduced customer traffic"
                        )

                    findings["potential_causes"] = potential_causes

                    insights.append(
                        f"⚠️ ALERT: Significant revenue drop of {revenue_change_pct:.1f}% detected!"
                    )
                    for cause in potential_causes:
                        insights.append(f"  - Possible cause: {cause}")

                    # Add recommendation for investigation
                    recommendations.append(
                        AgentRecommendation(
                            action_type="investigate_revenue_drop",
                            payload={
                                "period": str(latest.get("bucket")),
                                "drop_percentage": revenue_change_pct,
                                "potential_causes": potential_causes,
                            },
                            reasoning=f"Revenue dropped {revenue_change_pct:.1f}% which exceeds the 10% threshold",
                            requires_approval=False,
                        )
                    )

                elif revenue_change_pct > 10:
                    insights.append(
                        f"📈 Positive trend: Revenue increased by {revenue_change_pct:.1f}%"
                    )

            # Calculate averages for context
            if len(rows) >= 2:
                total_revenue = sum(float(r.get("revenue", 0)) for r in rows)
                avg_revenue = total_revenue / len(rows)
                findings["average_daily_revenue"] = round(avg_revenue, 2)
                insights.append(
                    f"Average daily revenue over {window_days} days: ${avg_revenue:,.2f}"
                )

        else:
            insights.append("No sales data found for the specified period.")

        return self.success(
            findings=findings,
            insights=insights,
            recommendations=recommendations,
        )
