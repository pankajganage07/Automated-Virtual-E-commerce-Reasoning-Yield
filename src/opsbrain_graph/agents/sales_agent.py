"""
SalesAgent: Analyzes sales and revenue data.

Slimmed architecture (2 core tools):
1. get_sales_summary - Aggregated sales metrics with trends
2. get_top_products - Best selling products

Complex queries (compare periods, regional, channel, product contribution)
return "cannot handle" and should route to DataAnalystAgent with HITL.
"""

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
    GetTopProductsRequest,
    GetSalesSummaryRequest,
)

logger = logging.getLogger("agent.sales")


class SalesAgent(BaseAgent):
    name = "sales"
    description = "Analyzes revenue, trends, and top products."

    metadata = AgentMetadata(
        name="sales",
        display_name="SALES",
        description=(
            "Analyzes sales performance with 2 core capabilities: "
            "sales summary with trends, and top selling products. "
            "Complex queries (compare periods, regional, channel analysis) "
            "should be routed to the Data Analyst."
        ),
        capabilities=[
            AgentCapability(
                name="summary",
                description="Get aggregated sales metrics with trend analysis",
                parameters={
                    "window_days": "Number of days to analyze (default: 7)",
                    "group_by": "Grouping: 'day' or 'week' (default: 'day')",
                },
                example_queries=[
                    "How are sales trending this week?",
                    "Show me revenue trends for the last 30 days",
                    "What's our sales performance?",
                ],
            ),
            AgentCapability(
                name="top_products",
                description="Find best-selling products by revenue",
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
        keywords=["sale", "revenue", "trend", "top", "product", "best", "money", "earning"],
        priority_boost=["revenue", "sales"],
    )

    # Patterns that indicate queries we can't handle (route to analyst)
    COMPLEX_QUERY_PATTERNS = [
        "compare",
        "yesterday",
        "last week",
        "vs",
        "versus",
        "region",
        "regional",
        "geography",
        "location",
        "channel",
        "mobile",
        "web",
        "marketplace",
        "contribution",
        "contributed",
        "caused",
        "driving",
    ]

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "summary")
        original_query = params.get("original_query", "")

        # Check if this is a complex query we can't handle
        if self._is_complex_query(original_query):
            return self._cannot_handle(original_query)

        if mode == "top_products":
            return await self._run_top_products(params)
        else:
            return await self._run_summary(params)

    def _is_complex_query(self, query: str) -> bool:
        """Check if the query requires complex analysis (route to analyst)."""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.COMPLEX_QUERY_PATTERNS)

    def _cannot_handle(self, query: str) -> AgentResult:
        """Return result indicating this query needs the Data Analyst."""
        return self.success(
            findings={
                "status": "cannot_handle",
                "reason": "Query requires complex analysis beyond core sales tools",
                "original_query": query,
                "suggestion": "Route to data_analyst agent for custom SQL with HITL approval",
            },
            insights=[
                "This query requires complex analysis (comparison, regional, or channel breakdown).",
                "Routing to Data Analyst agent for custom SQL analysis with HITL approval.",
            ],
        )

    async def _run_summary(self, params: dict[str, Any]) -> AgentResult:
        """Get sales summary with trend analysis."""
        window_days = params.get("window_days", 7)
        group_by = params.get("group_by", "day")

        try:
            resp = await self.tools.sales.get_sales_summary(
                GetSalesSummaryRequest(window_days=window_days, group_by=group_by)
            )
        except Exception as exc:
            logger.exception("sales agent (summary) failed: %s", exc)
            return self.failure(exc)

        summary = resp.summary
        trend = resp.trend
        trend_analysis = resp.trend_analysis

        findings: dict[str, Any] = {
            "summary": summary,
            "trend": trend,
            "trend_analysis": trend_analysis,
            "window_days": window_days,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        # Generate insights
        if summary:
            total_revenue = summary.get("total_revenue", 0)
            total_units = summary.get("total_units", 0)
            total_orders = summary.get("total_orders", 0)

            insights.append(f"Sales summary for the last {window_days} days:")
            insights.append(f"  Total revenue: ${total_revenue:,.2f}")
            insights.append(f"  Total units sold: {total_units:,}")
            insights.append(f"  Total orders: {total_orders:,}")

            if total_orders > 0:
                avg_order_value = total_revenue / total_orders
                insights.append(f"  Average order value: ${avg_order_value:,.2f}")

        # Trend analysis
        if trend_analysis == "increasing":
            insights.append("📈 Trend: Revenue is increasing compared to previous period")
        elif trend_analysis == "decreasing":
            insights.append("📉 Trend: Revenue is decreasing compared to previous period")
            recommendations.append(
                AgentRecommendation(
                    action_type="investigate_decline",
                    payload={"window_days": window_days, "trend": trend_analysis},
                    reasoning="Revenue trend is decreasing - recommend investigation",
                    requires_approval=False,
                )
            )
        else:
            insights.append("➡️ Trend: Revenue is stable")

        return self.success(
            findings=findings,
            insights=insights,
            recommendations=recommendations,
        )

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
            "total_top_products_revenue": resp.total_top_products_revenue,
        }
        insights: list[str] = []

        if products:
            insights.append(f"Top {len(products)} selling products in the last {window_days} days:")
            for i, product in enumerate(products, 1):
                category = f" ({product.category})" if product.category else ""
                insights.append(
                    f"  {i}. {product.name}{category} - ${product.revenue:,.2f} revenue, "
                    f"{product.units_sold} units sold"
                )

            insights.append(
                f"Total revenue from top {len(products)} products: "
                f"${resp.total_top_products_revenue:,.2f}"
            )
        else:
            insights.append(f"No product sales data found for the last {window_days} days.")

        return self.success(findings=findings, insights=insights)
