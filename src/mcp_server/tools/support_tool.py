"""
MCP Tools for customer support domain.

Core tools (2 main):
1. get_support_sentiment - Aggregate sentiment metrics for support tickets
2. get_ticket_trends - Analyze ticket trends over time

Complex queries (common issues, complaint comparisons)
should be routed to the Data Analyst agent with HITL approval.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import text
from typing import Any

from mcp_server.tools.base import BaseTool


# =============================================================================
# GET SUPPORT SENTIMENT TOOL
# =============================================================================


class SupportSentimentPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    product_id: int | None = Field(default=None, description="Optional product filter")
    issue_category: str | None = Field(default=None, description="Optional category filter")


class GetSupportSentimentTool(BaseTool):
    """Return aggregate sentiment metrics for support tickets."""

    name = "get_support_sentiment"

    def request_model(self) -> type[BaseModel]:
        return SupportSentimentPayload

    async def run(self, session, payload: SupportSentimentPayload) -> dict[str, Any]:
        statement = """
            SELECT
                COUNT(*) AS total,
                AVG(sentiment) AS avg_sentiment,
                SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END) AS negative_count,
                SUM(CASE WHEN sentiment >= 0.4 AND sentiment < 0.7 THEN 1 ELSE 0 END) AS neutral_count,
                SUM(CASE WHEN sentiment >= 0.7 THEN 1 ELSE 0 END) AS positive_count
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :window_days || ' days'
              AND (:product_id IS NULL OR product_id = :product_id)
              AND (:issue_category IS NULL OR issue_category = :issue_category)
        """
        result = await session.execute(
            text(statement),
            {
                "window_days": payload.window_days,
                "product_id": payload.product_id,
                "issue_category": payload.issue_category,
            },
        )
        row = result.one()

        total = row.total or 0
        negative_count = row.negative_count or 0

        return {
            "window_days": payload.window_days,
            "sentiment": {
                "avg_sentiment": round(float(row.avg_sentiment or 0), 2),
                "negative_ratio": round(negative_count / total, 2) if total > 0 else 0,
                "positive_count": row.positive_count or 0,
                "neutral_count": row.neutral_count or 0,
                "negative_count": negative_count,
            },
            "ticket_volume": total,
            "has_sentiment_issues": (negative_count / total > 0.3) if total > 0 else False,
        }


# =============================================================================
# GET TICKET TRENDS TOOL
# =============================================================================


class TicketTrendsPayload(BaseModel):
    window_days: int = Field(default=14, ge=1, le=90, description="Analysis window in days")
    group_by: str = Field(
        default="issue_category", description="Group by: issue_category, product, day"
    )
    product_id: int | None = Field(None, description="Optional product filter")


class GetTicketTrendsTool(BaseTool):
    """
    Analyze support ticket trends over time.
    Groups tickets by category, product, or day to identify patterns.
    """

    name = "get_ticket_trends"

    def request_model(self) -> type[BaseModel]:
        return TicketTrendsPayload

    async def run(self, session, payload: TicketTrendsPayload) -> dict[str, Any]:
        # Build query based on group_by
        group_mapping = {
            "issue_category": ("issue_category", "issue_category"),
            "product": ("product_id", "product_id"),
            "day": ("DATE(created_at)", "day"),
        }
        group_col, key_col = group_mapping.get(
            payload.group_by, ("issue_category", "issue_category")
        )

        # Get current period data
        current_stmt = text(
            f"""
            SELECT
                {group_col} AS group_key,
                COUNT(*) AS volume,
                AVG(sentiment) AS avg_sentiment,
                SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END) AS negative_count
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :window_days || ' days'
              AND (:product_id IS NULL OR product_id = :product_id)
            GROUP BY {group_col}
            ORDER BY volume DESC
        """
        )
        current_result = await session.execute(
            current_stmt, {"window_days": payload.window_days, "product_id": payload.product_id}
        )
        current_rows = list(current_result)

        # Get previous period data for comparison
        prev_stmt = text(
            f"""
            SELECT
                {group_col} AS group_key,
                COUNT(*) AS volume
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :prev_start || ' days'
              AND created_at < NOW() - INTERVAL :window_days || ' days'
              AND (:product_id IS NULL OR product_id = :product_id)
            GROUP BY {group_col}
        """
        )
        prev_result = await session.execute(
            prev_stmt,
            {
                "window_days": payload.window_days,
                "prev_start": payload.window_days * 2,
                "product_id": payload.product_id,
            },
        )
        prev_volumes = {row.group_key: row.volume for row in prev_result}

        # Build trend data
        trends = []
        total_volume = 0
        alerts = []

        for row in current_rows:
            volume = row.volume
            total_volume += volume
            prev_vol = prev_volumes.get(row.group_key, 0)

            # Calculate change percentage
            if prev_vol > 0:
                change_pct = ((volume - prev_vol) / prev_vol) * 100
            elif volume > 0:
                change_pct = 100.0  # New category
            else:
                change_pct = 0.0

            # Determine trend direction
            if change_pct > 20:
                trend = "increasing"
            elif change_pct < -20:
                trend = "decreasing"
            else:
                trend = "stable"

            key_value = str(row.group_key) if payload.group_by == "day" else row.group_key
            avg_sentiment = float(row.avg_sentiment or 0)

            trends.append(
                {
                    "key": key_value,
                    "volume": volume,
                    "previous_volume": prev_vol,
                    "change_pct": round(change_pct, 1),
                    "trend": trend,
                    "avg_sentiment": round(avg_sentiment, 2),
                    "negative_count": row.negative_count or 0,
                }
            )

            # Find notable patterns
            if change_pct > 50:
                alerts.append(f"Spike in '{key_value}': +{change_pct:.0f}% vs previous period")
            if avg_sentiment < 0.3:
                alerts.append(f"Low sentiment in '{key_value}': {avg_sentiment:.2f}")

        return {
            "window_days": payload.window_days,
            "group_by": payload.group_by,
            "total_volume": total_volume,
            "trends": trends,
            "alerts": alerts,
            "has_alerts": len(alerts) > 0,
        }
