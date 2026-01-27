from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import text
from typing import Any

from mcp_server.tools.base import BaseTool


class SupportSentimentPayload(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    product_id: int | None = None


class GetSupportSentimentTool(BaseTool):
    name = "get_support_sentiment"

    def request_model(self) -> type[BaseModel]:
        return SupportSentimentPayload

    async def run(self, session, payload: SupportSentimentPayload) -> dict[str, Any]:
        statement = """
            SELECT
                COUNT(*) AS total,
                AVG(sentiment) AS avg_sentiment,
                SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END)::float / COUNT(*) AS negative_ratio
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :window_days || ' days'
              AND (:product_id IS NULL OR product_id = :product_id)
        """
        result = await session.execute(
            text(statement),
            {"window_days": payload.window_days, "product_id": payload.product_id},
        )
        row = result.one()
        return {
            "sentiment": {
                "avg_sentiment": float(row.avg_sentiment or 0),
                "negative_ratio": float(row.negative_ratio or 0),
                "ticket_volume": row.total,
            }
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
        if payload.group_by == "issue_category":
            group_col = "issue_category"
            key_col = "issue_category"
        elif payload.group_by == "product":
            group_col = "product_id"
            key_col = "product_id"
        elif payload.group_by == "day":
            group_col = "DATE(created_at)"
            key_col = "day"
        else:
            group_col = "issue_category"
            key_col = "issue_category"

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

            trends.append(
                {
                    "key": key_value,
                    "volume": volume,
                    "previous_volume": prev_vol,
                    "change_pct": round(change_pct, 1),
                    "trend": trend,
                    "avg_sentiment": round(float(row.avg_sentiment or 0), 2),
                    "negative_count": row.negative_count,
                }
            )

        # Find notable patterns
        alerts = []
        for trend in trends:
            if trend["change_pct"] > 50:
                alerts.append(
                    f"Spike in '{trend['key']}': +{trend['change_pct']:.0f}% vs previous period"
                )
            if trend["avg_sentiment"] < 0.3:
                alerts.append(f"Low sentiment in '{trend['key']}': {trend['avg_sentiment']:.2f}")

        return {
            "window_days": payload.window_days,
            "group_by": payload.group_by,
            "total_volume": total_volume,
            "trends": trends,
            "alerts": alerts,
        }
