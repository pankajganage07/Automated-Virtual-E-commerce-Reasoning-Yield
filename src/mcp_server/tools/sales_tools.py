"""
MCP Tools for sales / orders domain.

Core tools (2 main):
1. get_sales_summary - Aggregated sales metrics with trend data
2. get_top_products - Best selling products

Complex queries (regional, channel, compare periods, product contribution)
should be routed to the Data Analyst agent with HITL approval.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


# =============================================================================
# GET SALES SUMMARY TOOL
# =============================================================================


class SalesSummaryPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    group_by: str = Field(default="day")  # day, week


class GetSalesSummaryTool(BaseTool):
    """Return revenue, units sold, order count, and daily trend for a time window."""

    name = "get_sales_summary"

    def request_model(self) -> type[BaseModel]:
        return SalesSummaryPayload

    async def run(self, session, payload: SalesSummaryPayload) -> dict[str, Any]:
        group_func = {
            "day": "date_trunc('day', timestamp)",
            "week": "date_trunc('week', timestamp)",
        }.get(payload.group_by, "date_trunc('day', timestamp)")

        statement = f"""
            SELECT
                {group_func} AS bucket,
                SUM(revenue) AS revenue,
                SUM(qty) AS units,
                COUNT(*) AS order_count
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL '{payload.window_days} days'
            GROUP BY bucket
            ORDER BY bucket DESC
        """
        result = await session.execute(text(statement))
        rows = [
            {
                "bucket": row.bucket.isoformat() if row.bucket else None,
                "revenue": float(row.revenue),
                "units": row.units,
                "order_count": row.order_count,
            }
            for row in result
        ]

        total_revenue = sum(r["revenue"] for r in rows)
        total_units = sum(r["units"] for r in rows)
        total_orders = sum(r["order_count"] for r in rows)

        # Calculate trend analysis
        trend_analysis = "stable"
        if len(rows) >= 2:
            recent = rows[0]["revenue"] if rows else 0
            previous = rows[1]["revenue"] if len(rows) > 1 else 0
            if previous > 0:
                change_pct = ((recent - previous) / previous) * 100
                if change_pct > 10:
                    trend_analysis = "increasing"
                elif change_pct < -10:
                    trend_analysis = "decreasing"

        return {
            "summary": {
                "total_revenue": round(total_revenue, 2),
                "total_units": total_units,
                "total_orders": total_orders,
                "window_days": payload.window_days,
            },
            "trend": rows,
            "trend_analysis": trend_analysis,
        }


# =============================================================================
# GET TOP PRODUCTS TOOL
# =============================================================================


class TopProductsPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=5, ge=1, le=50)


class GetTopProductsTool(BaseTool):
    """Return the top N products by revenue for a given window."""

    name = "get_top_products"

    def request_model(self) -> type[BaseModel]:
        return TopProductsPayload

    async def run(self, session, payload: TopProductsPayload) -> dict[str, Any]:
        statement = f"""
            SELECT
                p.id,
                p.name,
                p.category,
                SUM(o.qty) AS units_sold,
                SUM(o.revenue) AS revenue
            FROM products p
            JOIN orders o ON o.product_id = p.id
            WHERE o.timestamp >= NOW() - INTERVAL '{payload.window_days} days'
            GROUP BY p.id, p.name, p.category
            ORDER BY revenue DESC
            LIMIT :limit
        """
        result = await session.execute(
            text(statement),
            {"limit": payload.limit},
        )
        rows = [
            {
                "product_id": row.id,
                "name": row.name,
                "category": row.category,
                "units_sold": row.units_sold,
                "revenue": float(row.revenue),
            }
            for row in result
        ]

        total_revenue = sum(r["revenue"] for r in rows)

        return {
            "products": rows,
            "window_days": payload.window_days,
            "total_top_products_revenue": round(total_revenue, 2),
        }
