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
        statement = f"""
            SELECT
                p.id,
                p.name,
                SUM(o.qty) AS units_sold,
                SUM(o.revenue) AS revenue
            FROM products p
            JOIN orders o ON o.product_id = p.id
            WHERE o.timestamp >= NOW() - INTERVAL '{payload.window_days} days'
            GROUP BY p.id, p.name
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
                "units_sold": row.units_sold,
                "revenue": float(row.revenue),
            }
            for row in result
        ]
        return {"products": rows}


# =============================================================================
# COMPARE SALES PERIODS TOOL
# =============================================================================


class CompareSalesPeriodsPayload(BaseModel):
    """Compare two time periods for sales analysis."""

    current_days: int = Field(
        default=1, ge=1, le=30, description="Current period in days (e.g., 1 for yesterday)"
    )
    comparison_days: int = Field(
        default=7, ge=1, le=90, description="Comparison period in days (e.g., 7 for last week avg)"
    )
    group_by: str = Field(default="total", description="Group by: total, product, region, channel")


class CompareSalesPeriodsTool(BaseTool):
    """
    Compare sales between two periods to identify changes.
    Useful for questions like "Compare yesterday's sales with last week."
    """

    name = "compare_sales_periods"

    def request_model(self) -> type[BaseModel]:
        return CompareSalesPeriodsPayload

    async def run(self, session, payload: CompareSalesPeriodsPayload) -> dict[str, Any]:
        # Get current period metrics
        current_stmt = text(
            """
    SELECT
        COALESCE(SUM(revenue), 0) AS revenue,
        COALESCE(SUM(qty), 0) AS units,
        COUNT(*) AS order_count,
        COALESCE(AVG(revenue), 0) AS avg_order_value
    FROM orders
    WHERE timestamp >= NOW() - (:current_days * INTERVAL '1 day')
"""
        )
        current_result = await session.execute(current_stmt, {"current_days": payload.current_days})
        current = current_result.one()

        # Get comparison period metrics (previous period of same length for fair comparison)
        comparison_stmt = text(
            """
    SELECT
        COALESCE(SUM(revenue), 0) / NULLIF(:comparison_days, 0) AS daily_avg_revenue,
        COALESCE(SUM(qty), 0) / NULLIF(:comparison_days, 0) AS daily_avg_units,
        COUNT(*) / NULLIF(:comparison_days, 0) AS daily_avg_orders,
        COALESCE(AVG(revenue), 0) AS avg_order_value
    FROM orders
    WHERE timestamp >= NOW() - (:offset_days * INTERVAL '1 day')
      AND timestamp < NOW() - (:current_days * INTERVAL '1 day')
"""
        )
        comparison_result = await session.execute(
            comparison_stmt,
            {
                "current_days": payload.current_days,
                "comparison_days": payload.comparison_days,
                "offset_days": payload.current_days + payload.comparison_days,
            },
        )
        comparison = comparison_result.one()

        # Calculate expected vs actual
        expected_revenue = float(comparison.daily_avg_revenue or 0) * payload.current_days
        actual_revenue = float(current.revenue or 0)
        expected_units = float(comparison.daily_avg_units or 0) * payload.current_days
        actual_units = int(current.units or 0)

        # Calculate variances
        revenue_variance = actual_revenue - expected_revenue
        revenue_variance_pct = (
            (revenue_variance / expected_revenue * 100) if expected_revenue > 0 else 0
        )
        units_variance = actual_units - int(expected_units)
        units_variance_pct = (units_variance / expected_units * 100) if expected_units > 0 else 0

        # Analyze cause of variance
        analysis = []
        if revenue_variance_pct < -10:
            if units_variance_pct < revenue_variance_pct:
                analysis.append("Volume decline (fewer orders) is the primary driver")
            elif units_variance_pct > revenue_variance_pct:
                analysis.append(
                    "Lower order value is the primary driver (possible discounting or product mix)"
                )

            # Check if it's orders or units per order
            current_aov = float(current.avg_order_value or 0)
            comparison_aov = float(comparison.avg_order_value or 0)
            if current_aov < comparison_aov * 0.9:
                analysis.append(
                    f"Average order value dropped from ${comparison_aov:.2f} to ${current_aov:.2f}"
                )

        return {
            "current_period": {
                "days": payload.current_days,
                "revenue": round(actual_revenue, 2),
                "units": actual_units,
                "order_count": current.order_count,
                "avg_order_value": round(float(current.avg_order_value or 0), 2),
            },
            "comparison_period": {
                "days": payload.comparison_days,
                "daily_avg_revenue": round(float(comparison.daily_avg_revenue or 0), 2),
                "daily_avg_units": round(float(comparison.daily_avg_units or 0), 2),
                "expected_revenue": round(expected_revenue, 2),
                "expected_units": round(expected_units, 0),
            },
            "variance": {
                "revenue": round(revenue_variance, 2),
                "revenue_pct": round(revenue_variance_pct, 1),
                "units": units_variance,
                "units_pct": round(units_variance_pct, 1),
                "is_significant": abs(revenue_variance_pct) > 10,
            },
            "analysis": analysis,
        }


# =============================================================================
# REGIONAL SALES TOOL
# =============================================================================


class RegionalSalesPayload(BaseModel):
    """Analyze sales by region."""

    window_days: int = Field(default=7, ge=1, le=90)
    compare_to_previous: bool = Field(default=True, description="Compare to previous period")


class GetRegionalSalesTool(BaseTool):
    """
    Analyze sales performance by geographic region.
    Identifies which regions are underperforming.
    """

    name = "get_regional_sales"

    def request_model(self) -> type[BaseModel]:
        return RegionalSalesPayload

    async def run(self, session, payload: RegionalSalesPayload) -> dict[str, Any]:
        # Current period by region
        current_stmt = text(
            """
    SELECT
        region,
        COALESCE(SUM(revenue), 0) AS revenue,
        COALESCE(SUM(qty), 0) AS units,
        COUNT(*) AS order_count
    FROM orders
    WHERE timestamp >= NOW() - (:window_days * INTERVAL '1 day')
    GROUP BY region
    ORDER BY revenue DESC
"""
        )
        current_result = await session.execute(current_stmt, {"window_days": payload.window_days})
        current_rows = list(current_result)

        # Previous period for comparison
        prev_data = {}
        if payload.compare_to_previous:
            prev_stmt = text(
                """
    SELECT
        region,
        COALESCE(SUM(revenue), 0) AS revenue,
        COALESCE(SUM(qty), 0) AS units
    FROM orders
    WHERE timestamp >= NOW() - (:offset_days * INTERVAL '1 day')
      AND timestamp < NOW() - (:window_days * INTERVAL '1 day')
    GROUP BY region
"""
            )
            prev_result = await session.execute(
                prev_stmt,
                {"window_days": payload.window_days, "offset_days": payload.window_days * 2},
            )
            prev_data = {
                row.region: {"revenue": float(row.revenue), "units": row.units}
                for row in prev_result
            }

        regions = []
        underperforming = []
        total_revenue = sum(float(row.revenue) for row in current_rows)

        for row in current_rows:
            revenue = float(row.revenue)
            prev = prev_data.get(row.region, {"revenue": 0, "units": 0})
            prev_revenue = prev["revenue"]

            change_pct = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0

            region_data = {
                "region": row.region,
                "revenue": round(revenue, 2),
                "units": row.units,
                "order_count": row.order_count,
                "revenue_share_pct": (
                    round(revenue / total_revenue * 100, 1) if total_revenue > 0 else 0
                ),
                "change_pct": round(change_pct, 1),
            }
            regions.append(region_data)

            if change_pct < -15:
                underperforming.append(region_data)

        return {
            "window_days": payload.window_days,
            "total_revenue": round(total_revenue, 2),
            "regions": regions,
            "underperforming_regions": underperforming,
            "has_underperforming": len(underperforming) > 0,
        }


# =============================================================================
# CHANNEL PERFORMANCE TOOL
# =============================================================================


class ChannelPerformancePayload(BaseModel):
    """Analyze sales by channel."""

    window_days: int = Field(default=7, ge=1, le=90)


class GetChannelPerformanceTool(BaseTool):
    """
    Analyze sales performance by channel (e.g., web, mobile, marketplace).
    """

    name = "get_channel_performance"

    def request_model(self) -> type[BaseModel]:
        return ChannelPerformancePayload

    async def run(self, session, payload: ChannelPerformancePayload) -> dict[str, Any]:
        # Current period by channel
        current_stmt = text(
            """
            SELECT
                channel,
                COALESCE(SUM(revenue), 0) AS revenue,
                COALESCE(SUM(qty), 0) AS units,
                COUNT(*) AS order_count,
                COALESCE(AVG(revenue / NULLIF(qty, 0)), 0) AS avg_unit_price
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :window_days || ' days'
            GROUP BY channel
            ORDER BY revenue DESC
        """
        )
        current_result = await session.execute(current_stmt, {"window_days": payload.window_days})
        current_rows = list(current_result)

        # Previous period for comparison
        prev_stmt = text(
            """
            SELECT
                channel,
                COALESCE(SUM(revenue), 0) AS revenue
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :offset_days || ' days'
              AND timestamp < NOW() - INTERVAL :window_days || ' days'
            GROUP BY channel
        """
        )
        prev_result = await session.execute(
            prev_stmt, {"window_days": payload.window_days, "offset_days": payload.window_days * 2}
        )
        prev_data = {row.channel: float(row.revenue) for row in prev_result}

        channels = []
        worst_channel = None
        worst_change = 0
        total_revenue = sum(float(row.revenue) for row in current_rows)

        for row in current_rows:
            revenue = float(row.revenue)
            prev_revenue = prev_data.get(row.channel, 0)
            change_pct = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0

            channel_data = {
                "channel": row.channel,
                "revenue": round(revenue, 2),
                "units": row.units,
                "order_count": row.order_count,
                "avg_unit_price": round(float(row.avg_unit_price or 0), 2),
                "revenue_share_pct": (
                    round(revenue / total_revenue * 100, 1) if total_revenue > 0 else 0
                ),
                "change_pct": round(change_pct, 1),
            }
            channels.append(channel_data)

            if change_pct < worst_change:
                worst_change = change_pct
                worst_channel = channel_data

        return {
            "window_days": payload.window_days,
            "total_revenue": round(total_revenue, 2),
            "channels": channels,
            "worst_performing": worst_channel,
        }


# =============================================================================
# PRODUCT CONTRIBUTION ANALYSIS TOOL
# =============================================================================


class ProductContributionPayload(BaseModel):
    """Analyze which products contributed to revenue changes."""

    current_days: int = Field(default=1, ge=1, le=7)
    comparison_days: int = Field(default=7, ge=1, le=30)
    limit: int = Field(default=10, ge=1, le=50)


class GetProductContributionTool(BaseTool):
    """
    Identify which products contributed most to revenue changes.
    Useful for "Which products contributed to the revenue drop?"
    """

    name = "get_product_contribution"

    def request_model(self) -> type[BaseModel]:
        return ProductContributionPayload

    async def run(self, session, payload: ProductContributionPayload) -> dict[str, Any]:
        # Get product performance for both periods
        stmt = text(
            """
    WITH current_period AS (
        SELECT
            p.id,
            p.name,
            COALESCE(SUM(o.revenue), 0) AS revenue,
            COALESCE(SUM(o.qty), 0) AS units
        FROM products p
        LEFT JOIN orders o ON o.product_id = p.id
            AND o.timestamp >= NOW() - (:current_days * INTERVAL '1 day')
        GROUP BY p.id, p.name
    ),
    comparison_period AS (
        SELECT
            p.id,
            COALESCE(SUM(o.revenue), 0) / NULLIF(:comparison_days, 0) * :current_days AS expected_revenue,
            COALESCE(SUM(o.qty), 0) / NULLIF(:comparison_days, 0) * :current_days AS expected_units
        FROM products p
        LEFT JOIN orders o ON o.product_id = p.id
            AND o.timestamp >= NOW() - (:offset_days * INTERVAL '1 day')
            AND o.timestamp < NOW() - (:current_days * INTERVAL '1 day')
        GROUP BY p.id
    )
    SELECT
        c.id,
        c.name,
        c.revenue AS actual_revenue,
        c.units AS actual_units,
        COALESCE(cp.expected_revenue, 0) AS expected_revenue,
        COALESCE(cp.expected_units, 0) AS expected_units,
        c.revenue - COALESCE(cp.expected_revenue, 0) AS revenue_impact
    FROM current_period c
    LEFT JOIN comparison_period cp ON cp.id = c.id
    ORDER BY revenue_impact ASC
    LIMIT :limit
"""
        )
        result = await session.execute(
            stmt,
            {
                "current_days": payload.current_days,
                "comparison_days": payload.comparison_days,
                "offset_days": payload.current_days + payload.comparison_days,
                "limit": payload.limit,
            },
        )

        products = []
        total_negative_impact = 0
        for row in result:
            impact = float(row.revenue_impact)
            products.append(
                {
                    "product_id": row.id,
                    "name": row.name,
                    "actual_revenue": round(float(row.actual_revenue), 2),
                    "expected_revenue": round(float(row.expected_revenue), 2),
                    "revenue_impact": round(impact, 2),
                    "actual_units": row.actual_units,
                    "expected_units": round(float(row.expected_units), 0),
                }
            )
            if impact < 0:
                total_negative_impact += abs(impact)

        return {
            "current_days": payload.current_days,
            "comparison_days": payload.comparison_days,
            "products_by_impact": products,
            "total_negative_impact": round(total_negative_impact, 2),
        }
