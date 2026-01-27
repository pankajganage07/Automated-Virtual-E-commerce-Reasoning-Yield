from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


class SalesSummaryPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    group_by: str = Field(default="day")  # day, week


class GetSalesSummaryTool(BaseTool):
    name = "get_sales_summary"

    def request_model(self) -> type[BaseModel]:
        return SalesSummaryPayload

    async def run(self, session, payload: SalesSummaryPayload) -> dict[str, Any]:
        group_func = {
            "day": "date_trunc('day', timestamp)",
            "week": "date_trunc('week', timestamp)",
        }[payload.group_by]
        statement = f"""
            SELECT
                {group_func} AS bucket,
                SUM(revenue) AS revenue,
                SUM(qty) AS units
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
            }
            for row in result
        ]

        total_revenue = sum(r["revenue"] for r in rows)
        total_units = sum(r["units"] for r in rows)

        return {
            "summary": {"total_revenue": total_revenue, "total_units": total_units},
            "trend": rows,
        }


class TopProductsPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=5, ge=1, le=50)


class GetTopProductsTool(BaseTool):
    name = "get_top_products"

    def request_model(self) -> type[BaseModel]:
        return TopProductsPayload

    async def run(self, session, payload: TopProductsPayload) -> dict[str, Any]:
        statement = """
            SELECT
                p.id,
                p.name,
                SUM(o.qty) AS units_sold,
                SUM(o.revenue) AS revenue
            FROM products p
            JOIN orders o ON o.product_id = p.id
            WHERE o.timestamp >= NOW() - INTERVAL :window_days || ' days'
            GROUP BY p.id, p.name
            ORDER BY revenue DESC
            LIMIT :limit
        """
        result = await session.execute(
            text(statement),
            {"window_days": payload.window_days, "limit": payload.limit},
        )
        rows = [
            {
                "product_id": row.id,
                "name": row.name,
                "units_sold": row.units_sold,
                "revenue": float(row.revenue),
            }
            for row in result
        ]
        return {"products": rows}
