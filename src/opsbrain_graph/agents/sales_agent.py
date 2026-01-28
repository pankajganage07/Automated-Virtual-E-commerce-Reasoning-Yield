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
    CompareSalesPeriodsRequest,
    GetRegionalSalesRequest,
    GetChannelPerformanceRequest,
    GetProductContributionRequest,
)

logger = logging.getLogger("agent.sales")


class SalesAgent(BaseAgent):
    name = "sales"
    description = "Analyzes revenue, trends, top products, and anomalies."

    metadata = AgentMetadata(
        name="sales",
        display_name="SALES",
        description="Analyzes revenue trends, sales performance, and detects anomalies. Can identify top-selling products, compare periods, analyze by region/channel, and identify revenue contributors.",
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
            AgentCapability(
                name="compare_periods",
                description="Compare sales between current and previous periods to identify changes",
                parameters={
                    "current_days": "Current period in days (default: 1)",
                    "previous_days": "Previous period in days for comparison (default: 7)",
                },
                example_queries=[
                    "Compare yesterday's sales with last week",
                    "How does this week compare to last week?",
                    "Did sales recover today vs yesterday?",
                ],
            ),
            AgentCapability(
                name="regional",
                description="Analyze sales performance by geographic region",
                parameters={
                    "window_days": "Time period to analyze (default: 7)",
                    "compare_to_avg": "Compare to historical average (default: true)",
                },
                example_queries=[
                    "Did any region perform worse than usual?",
                    "Which region had the best sales?",
                    "Regional sales breakdown",
                ],
            ),
            AgentCapability(
                name="channel",
                description="Analyze sales performance by sales channel (web, mobile, etc.)",
                parameters={
                    "window_days": "Time period to analyze (default: 7)",
                },
                example_queries=[
                    "Which channel performed the worst yesterday?",
                    "Channel performance breakdown",
                    "Are mobile sales up or down?",
                ],
            ),
            AgentCapability(
                name="product_contribution",
                description="Identify which products contributed most to revenue changes",
                parameters={
                    "current_days": "Current period in days (default: 1)",
                    "previous_days": "Previous period for comparison (default: 7)",
                    "limit": "Number of products to analyze (default: 10)",
                },
                example_queries=[
                    "Which products contributed most to the revenue drop?",
                    "What products are driving revenue growth?",
                    "Product-level revenue impact",
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
            "region",
            "channel",
            "compare",
        ],
        priority_boost=["revenue", "sales drop", "urgent"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "trends")

        if mode == "top_products":
            return await self._run_top_products(params)
        elif mode == "compare_periods":
            return await self._run_compare_periods(params)
        elif mode == "regional":
            return await self._run_regional(params)
        elif mode == "channel":
            return await self._run_channel(params)
        elif mode == "product_contribution":
            return await self._run_product_contribution(params)
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

    async def _run_compare_periods(self, params: dict[str, Any]) -> AgentResult:
        """Compare sales between two periods."""
        current_days = params.get("current_days", 1)
        previous_days = params.get("previous_days", 7)

        try:
            resp = await self.tools.sales.compare_sales_periods(
                CompareSalesPeriodsRequest(
                    current_days=current_days,
                    previous_days=previous_days,
                )
            )
        except Exception as exc:
            logger.exception("sales agent (compare_periods) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "current_period": {
                "days": current_days,
                "revenue": resp.current_revenue,
                "orders": resp.current_orders,
            },
            "previous_period": {
                "days": previous_days,
                "revenue": resp.previous_revenue,
                "orders": resp.previous_orders,
            },
            "changes": {
                "revenue_change_pct": resp.revenue_change_pct,
                "order_change_pct": resp.order_change_pct,
                "avg_order_value_change_pct": resp.avg_order_value_change_pct,
            },
            "trend": resp.trend,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"Comparing last {current_days} day(s) to previous {previous_days} days:")
        insights.append(f"  Current revenue: ${resp.current_revenue:,.2f}")
        insights.append(
            f"  Previous avg daily: ${resp.previous_revenue / previous_days if previous_days > 0 else 0:,.2f}"
        )
        insights.append(f"  Revenue change: {resp.revenue_change_pct:+.1f}%")
        insights.append(f"  Order change: {resp.order_change_pct:+.1f}%")
        insights.append(f"  Trend: {resp.trend}")

        if resp.revenue_change_pct < -15:
            insights.append(
                f"⚠️ ALERT: Significant revenue decline of {resp.revenue_change_pct:.1f}%"
            )
            recommendations.append(
                AgentRecommendation(
                    action_type="investigate_decline",
                    payload={"decline_pct": resp.revenue_change_pct},
                    reasoning="Revenue decline exceeds 15% threshold",
                    requires_approval=False,
                )
            )
        elif resp.revenue_change_pct > 15:
            insights.append(
                f"📈 Strong growth: Revenue increased by {resp.revenue_change_pct:.1f}%"
            )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_regional(self, params: dict[str, Any]) -> AgentResult:
        """Analyze sales by region."""
        window_days = params.get("window_days", 7)
        compare_to_avg = params.get("compare_to_avg", True)

        try:
            resp = await self.tools.sales.get_regional_sales(
                GetRegionalSalesRequest(
                    window_days=window_days,
                    compare_to_avg=compare_to_avg,
                )
            )
        except Exception as exc:
            logger.exception("sales agent (regional) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "window_days": window_days,
            "regions": [r.model_dump() for r in resp.regions],
            "underperforming_regions": [r.model_dump() for r in resp.underperforming_regions],
            "top_region": resp.top_region,
            "worst_region": resp.worst_region,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"Regional sales analysis for the last {window_days} days:")

        for region in resp.regions:
            trend_icon = "📈" if region.change_pct > 0 else "📉" if region.change_pct < 0 else "➡️"
            insights.append(
                f"  {trend_icon} {region.region}: ${region.revenue:,.2f} ({region.change_pct:+.1f}% vs avg)"
            )

        if resp.underperforming_regions:
            insights.append(f"⚠️ Underperforming regions: {len(resp.underperforming_regions)}")
            for r in resp.underperforming_regions:
                insights.append(f"    - {r.region}: {r.change_pct:.1f}% below average")

        if resp.top_region:
            insights.append(f"🏆 Top performing region: {resp.top_region}")
        if resp.worst_region:
            insights.append(f"⚠️ Worst performing region: {resp.worst_region}")

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_channel(self, params: dict[str, Any]) -> AgentResult:
        """Analyze sales by channel."""
        window_days = params.get("window_days", 7)

        try:
            resp = await self.tools.sales.get_channel_performance(
                GetChannelPerformanceRequest(window_days=window_days)
            )
        except Exception as exc:
            logger.exception("sales agent (channel) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "window_days": window_days,
            "channels": [c.model_dump() for c in resp.channels],
            "top_channel": resp.top_channel,
            "worst_channel": resp.worst_channel,
            "total_revenue": resp.total_revenue,
        }
        insights: list[str] = []

        insights.append(f"Channel performance for the last {window_days} days:")
        insights.append(f"Total revenue: ${resp.total_revenue:,.2f}")

        for channel in resp.channels:
            insights.append(
                f"  - {channel.channel}: ${channel.revenue:,.2f} "
                f"({channel.revenue_share:.1f}% of total), {channel.orders} orders"
            )

        if resp.top_channel:
            insights.append(f"🏆 Best channel: {resp.top_channel}")
        if resp.worst_channel:
            insights.append(f"⚠️ Worst channel: {resp.worst_channel}")

        return self.success(findings=findings, insights=insights)

    async def _run_product_contribution(self, params: dict[str, Any]) -> AgentResult:
        """Identify products contributing to revenue changes."""
        current_days = params.get("current_days", 1)
        previous_days = params.get("previous_days", 7)
        limit = params.get("limit", 10)

        try:
            resp = await self.tools.sales.get_product_contribution(
                GetProductContributionRequest(
                    current_days=current_days,
                    previous_days=previous_days,
                    limit=limit,
                )
            )
        except Exception as exc:
            logger.exception("sales agent (product_contribution) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "current_days": current_days,
            "previous_days": previous_days,
            "products": [p.model_dump() for p in resp.products],
            "biggest_gainers": [p.model_dump() for p in resp.biggest_gainers],
            "biggest_losers": [p.model_dump() for p in resp.biggest_losers],
            "overall_change_pct": resp.overall_change_pct,
        }
        insights: list[str] = []

        insights.append(
            f"Product revenue contribution (last {current_days} day(s) vs {previous_days} days):"
        )
        insights.append(f"Overall revenue change: {resp.overall_change_pct:+.1f}%")

        if resp.biggest_losers:
            insights.append("📉 Products with biggest revenue drops:")
            for p in resp.biggest_losers[:5]:
                insights.append(
                    f"  - {p.name}: {p.change_pct:+.1f}% (${p.current_revenue:,.2f} vs avg ${p.previous_avg_revenue:,.2f})"
                )

        if resp.biggest_gainers:
            insights.append("📈 Products with biggest revenue gains:")
            for p in resp.biggest_gainers[:5]:
                insights.append(
                    f"  - {p.name}: {p.change_pct:+.1f}% (${p.current_revenue:,.2f} vs avg ${p.previous_avg_revenue:,.2f})"
                )

        return self.success(findings=findings, insights=insights)
