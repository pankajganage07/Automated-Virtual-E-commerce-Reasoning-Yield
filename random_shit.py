"""
MCP Tools for sales / orders domain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_server.tools.base import BaseTool


class GetSalesSummaryTool(BaseTool):
    """Return revenue, units sold, and order count for a given time window."""

    name = "get_sales_summary"
    description = "Retrieve aggregated sales metrics (revenue, units, orders) for a specified number of past days."
    parameters = {
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of past days to aggregate (default 7).",
                "default": 7,
            }
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        window_days = int(args.get("window_days", 7))

        # Calculate the start date
        start_date = datetime.now(timezone.utc) - timedelta(days=window_days)

        query = text(
            """
            SELECT
                COALESCE(SUM(revenue), 0) AS total_revenue,
                COALESCE(SUM(qty), 0)     AS total_units,
                COUNT(*)                  AS order_count
            FROM orders
            WHERE timestamp >= :start_date
        """
        )

        result = await session.execute(query, {"start_date": start_date})
        row = result.mappings().one_or_none()

        if row is None:
            return {
                "total_revenue": 0,
                "total_units": 0,
                "order_count": 0,
                "window_days": window_days,
            }

        return {
            "total_revenue": float(row["total_revenue"]),
            "total_units": int(row["total_units"]),
            "order_count": int(row["order_count"]),
            "window_days": window_days,
        }


class GetTopProductsTool(BaseTool):
    """Return the top N products by revenue for a given window."""

    name = "get_top_products"
    description = "Retrieve the top-selling products ranked by revenue."
    parameters = {
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of past days to consider (default 7).",
                "default": 7,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of products to return (default 5).",
                "default": 5,
            },
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        window_days = int(args.get("window_days", 7))
        limit = int(args.get("limit", 5))

        # Calculate the start date
        start_date = datetime.now(timezone.utc) - timedelta(days=window_days)

        query = text(
            """
            SELECT
                p.id AS product_id,
                p.name AS product_name,
                SUM(o.qty) AS units_sold,
                SUM(o.revenue) AS revenue
            FROM orders o
            JOIN products p ON o.product_id = p.id
            WHERE o.timestamp >= :start_date
            GROUP BY p.id, p.name
            ORDER BY revenue DESC
            LIMIT :limit
        """
        )

        result = await session.execute(query, {"start_date": start_date, "limit": limit})
        rows = result.mappings().all()

        products = [
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "units_sold": int(row["units_sold"]),
                "revenue": float(row["revenue"]),
            }
            for row in rows
        ]

        return {"products": products, "window_days": window_days}


class GetDailySalesTrendTool(BaseTool):
    """Return daily sales trend data for a time window."""

    name = "get_daily_sales_trend"
    description = "Retrieve day-by-day sales trend data for analysis."
    parameters = {
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of past days to retrieve (default 7).",
                "default": 7,
            }
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        window_days = int(args.get("window_days", 7))

        # Calculate the start date
        start_date = datetime.now(timezone.utc) - timedelta(days=window_days)

        query = text(
            """
            SELECT
                DATE(timestamp) AS date,
                COALESCE(SUM(revenue), 0) AS revenue,
                COALESCE(SUM(qty), 0) AS units_sold,
                COUNT(*) AS order_count
            FROM orders
            WHERE timestamp >= :start_date
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """
        )

        result = await session.execute(query, {"start_date": start_date})
        rows = result.mappings().all()

        trend = [
            {
                "date": row["date"].isoformat() if row["date"] else None,
                "revenue": float(row["revenue"]),
                "units_sold": int(row["units_sold"]),
                "order_count": int(row["order_count"]),
            }
            for row in rows
        ]

        return {"trend": trend, "window_days": window_days}


class CompareSalesPeriodsTool(BaseTool):
    """Compare sales between current period and previous period."""

    name = "compare_sales_periods"
    description = "Compare sales metrics between a current period and a previous period."
    parameters = {
        "type": "object",
        "properties": {
            "current_days": {
                "type": "integer",
                "description": "Number of days for current period (default 1 = yesterday).",
                "default": 1,
            },
            "previous_days": {
                "type": "integer",
                "description": "Number of days for previous period to compare (default 7).",
                "default": 7,
            },
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        current_days = int(args.get("current_days", 1))
        previous_days = int(args.get("previous_days", 7))

        now = datetime.now(timezone.utc)

        # Current period: last N days
        current_start = now - timedelta(days=current_days)
        current_end = now

        # Previous period: the N days before current period
        previous_start = current_start - timedelta(days=previous_days)
        previous_end = current_start

        # Query for current period
        current_query = text(
            """
            SELECT
                COALESCE(SUM(revenue), 0) AS revenue,
                COALESCE(SUM(qty), 0) AS units,
                COUNT(*) AS order_count,
                COALESCE(AVG(revenue), 0) AS avg_order_value
            FROM orders
            WHERE timestamp >= :start_date AND timestamp < :end_date
        """
        )

        current_result = await session.execute(
            current_query, {"start_date": current_start, "end_date": current_end}
        )
        current_row = current_result.mappings().one()

        # Query for previous period
        previous_result = await session.execute(
            current_query, {"start_date": previous_start, "end_date": previous_end}
        )
        previous_row = previous_result.mappings().one()

        # Calculate percentage changes
        def calc_change(current: float, previous: float) -> float | None:
            if previous == 0:
                return None if current == 0 else 100.0
            return round(((current - previous) / previous) * 100, 2)

        current_revenue = float(current_row["revenue"])
        previous_revenue = float(previous_row["revenue"])
        current_units = int(current_row["units"])
        previous_units = int(previous_row["units"])
        current_orders = int(current_row["order_count"])
        previous_orders = int(previous_row["order_count"])

        return {
            "current_period": {
                "days": current_days,
                "start_date": current_start.date().isoformat(),
                "end_date": current_end.date().isoformat(),
                "revenue": current_revenue,
                "units": current_units,
                "order_count": current_orders,
                "avg_order_value": float(current_row["avg_order_value"]),
            },
            "previous_period": {
                "days": previous_days,
                "start_date": previous_start.date().isoformat(),
                "end_date": previous_end.date().isoformat(),
                "revenue": previous_revenue,
                "units": previous_units,
                "order_count": previous_orders,
                "avg_order_value": float(previous_row["avg_order_value"]),
            },
            "changes": {
                "revenue_pct": calc_change(current_revenue, previous_revenue),
                "units_pct": calc_change(current_units, previous_units),
                "orders_pct": calc_change(current_orders, previous_orders),
            },
        }


class GetRegionalSalesTool(BaseTool):
    """Analyze sales by region."""

    name = "get_regional_sales"
    description = "Get sales breakdown by region for a time window."
    parameters = {
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of past days to analyze (default 7).",
                "default": 7,
            },
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        window_days = int(args.get("window_days", 7))

        start_date = datetime.now(timezone.utc) - timedelta(days=window_days)

        query = text(
            """
            SELECT
                region,
                COALESCE(SUM(revenue), 0) AS revenue,
                COALESCE(SUM(qty), 0) AS units,
                COUNT(*) AS order_count
            FROM orders
            WHERE timestamp >= :start_date
            GROUP BY region
            ORDER BY revenue DESC
        """
        )

        result = await session.execute(query, {"start_date": start_date})
        rows = result.mappings().all()

        regions = [
            {
                "region": row["region"],
                "revenue": float(row["revenue"]),
                "units": int(row["units"]),
                "order_count": int(row["order_count"]),
            }
            for row in rows
        ]

        # Calculate totals for percentage
        total_revenue = sum(r["revenue"] for r in regions)
        for r in regions:
            r["revenue_pct"] = (
                round((r["revenue"] / total_revenue * 100), 2) if total_revenue > 0 else 0
            )

        return {"regions": regions, "window_days": window_days, "total_revenue": total_revenue}


class GetChannelPerformanceTool(BaseTool):
    """Analyze sales by channel."""

    name = "get_channel_performance"
    description = "Get sales breakdown by sales channel for a time window."
    parameters = {
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of past days to analyze (default 7).",
                "default": 7,
            },
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        window_days = int(args.get("window_days", 7))

        start_date = datetime.now(timezone.utc) - timedelta(days=window_days)

        query = text(
            """
            SELECT
                channel,
                COALESCE(SUM(revenue), 0) AS revenue,
                COALESCE(SUM(qty), 0) AS units,
                COUNT(*) AS order_count
            FROM orders
            WHERE timestamp >= :start_date
            GROUP BY channel
            ORDER BY revenue DESC
        """
        )

        result = await session.execute(query, {"start_date": start_date})
        rows = result.mappings().all()

        channels = [
            {
                "channel": row["channel"],
                "revenue": float(row["revenue"]),
                "units": int(row["units"]),
                "order_count": int(row["order_count"]),
            }
            for row in rows
        ]

        # Calculate totals for percentage
        total_revenue = sum(c["revenue"] for c in channels)
        for c in channels:
            c["revenue_pct"] = (
                round((c["revenue"] / total_revenue * 100), 2) if total_revenue > 0 else 0
            )

        return {"channels": channels, "window_days": window_days, "total_revenue": total_revenue}


class GetProductContributionTool(BaseTool):
    """Identify products contributing to revenue changes."""

    name = "get_product_contribution"
    description = "Identify which products contributed most to revenue changes between periods."
    parameters = {
        "type": "object",
        "properties": {
            "current_days": {
                "type": "integer",
                "description": "Number of days for current period (default 1).",
                "default": 1,
            },
            "previous_days": {
                "type": "integer",
                "description": "Number of days for previous period (default 7).",
                "default": 7,
            },
            "limit": {
                "type": "integer",
                "description": "Number of top products to return (default 10).",
                "default": 10,
            },
        },
        "required": [],
    }

    async def run(self, args: dict[str, Any], session: AsyncSession) -> dict[str, Any]:
        current_days = int(args.get("current_days", 1))
        previous_days = int(args.get("previous_days", 7))
        limit = int(args.get("limit", 10))

        now = datetime.now(timezone.utc)

        current_start = now - timedelta(days=current_days)
        current_end = now
        previous_start = current_start - timedelta(days=previous_days)
        previous_end = current_start

        # Get current period sales by product
        query = text(
            """
            WITH current_sales AS (
                SELECT
                    product_id,
                    COALESCE(SUM(revenue), 0) AS revenue,
                    COALESCE(SUM(qty), 0) AS units
                FROM orders
                WHERE timestamp >= :current_start AND timestamp < :current_end
                GROUP BY product_id
            ),
            previous_sales AS (
                SELECT
                    product_id,
                    COALESCE(SUM(revenue), 0) AS revenue,
                    COALESCE(SUM(qty), 0) AS units
                FROM orders
                WHERE timestamp >= :previous_start AND timestamp < :previous_end
                GROUP BY product_id
            )
            SELECT
                p.id AS product_id,
                p.name AS product_name,
                COALESCE(c.revenue, 0) AS current_revenue,
                COALESCE(c.units, 0) AS current_units,
                COALESCE(pr.revenue, 0) AS previous_revenue,
                COALESCE(pr.units, 0) AS previous_units,
                COALESCE(c.revenue, 0) - COALESCE(pr.revenue, 0) AS revenue_change
            FROM products p
            LEFT JOIN current_sales c ON p.id = c.product_id
            LEFT JOIN previous_sales pr ON p.id = pr.product_id
            WHERE COALESCE(c.revenue, 0) > 0 OR COALESCE(pr.revenue, 0) > 0
            ORDER BY ABS(COALESCE(c.revenue, 0) - COALESCE(pr.revenue, 0)) DESC
            LIMIT :limit
        """
        )

        result = await session.execute(
            query,
            {
                "current_start": current_start,
                "current_end": current_end,
                "previous_start": previous_start,
                "previous_end": previous_end,
                "limit": limit,
            },
        )
        rows = result.mappings().all()

        products = []
        for row in rows:
            current_rev = float(row["current_revenue"])
            previous_rev = float(row["previous_revenue"])

            if previous_rev > 0:
                change_pct = round(((current_rev - previous_rev) / previous_rev) * 100, 2)
            elif current_rev > 0:
                change_pct = 100.0
            else:
                change_pct = 0.0

            products.append(
                {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "current_revenue": current_rev,
                    "current_units": int(row["current_units"]),
                    "previous_revenue": previous_rev,
                    "previous_units": int(row["previous_units"]),
                    "revenue_change": float(row["revenue_change"]),
                    "change_pct": change_pct,
                }
            )

        return {
            "products": products,
            "current_period": {"days": current_days, "start": current_start.date().isoformat()},
            "previous_period": {"days": previous_days, "start": previous_start.date().isoformat()},
        }
