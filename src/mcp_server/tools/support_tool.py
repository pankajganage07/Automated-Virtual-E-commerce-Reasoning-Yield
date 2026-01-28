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


# =============================================================================
# GET COMMON ISSUES TOOL
# =============================================================================


class CommonIssuesPayload(BaseModel):
    """Find most common customer issues."""

    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=10, ge=1, le=50)


class GetCommonIssuesTool(BaseTool):
    """
    Identify the most common issues reported by customers.
    Answers: "Is there a common issue reported by customers?"
    """

    name = "get_common_issues"

    def request_model(self) -> type[BaseModel]:
        return CommonIssuesPayload

    async def run(self, session, payload: CommonIssuesPayload) -> dict[str, Any]:
        # Get issues grouped by category with sample descriptions
        stmt = text(
            """
            WITH issue_stats AS (
                SELECT
                    issue_category,
                    COUNT(*) AS volume,
                    AVG(sentiment) AS avg_sentiment,
                    SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END) AS negative_count
                FROM support_tickets
                WHERE created_at >= NOW() - INTERVAL :window_days || ' days'
                GROUP BY issue_category
            ),
            issue_samples AS (
                SELECT DISTINCT ON (issue_category)
                    issue_category,
                    description AS sample_description
                FROM support_tickets
                WHERE created_at >= NOW() - INTERVAL :window_days || ' days'
                ORDER BY issue_category, created_at DESC
            )
            SELECT 
                s.issue_category,
                s.volume,
                s.avg_sentiment,
                s.negative_count,
                i.sample_description
            FROM issue_stats s
            LEFT JOIN issue_samples i ON i.issue_category = s.issue_category
            ORDER BY s.volume DESC
            LIMIT :limit
        """
        )
        result = await session.execute(
            stmt, {"window_days": payload.window_days, "limit": payload.limit}
        )

        issues = []
        total_tickets = 0
        most_common = None

        for row in result:
            volume = row.volume
            total_tickets += volume

            issue = {
                "issue_category": row.issue_category,
                "volume": volume,
                "avg_sentiment": round(float(row.avg_sentiment or 0), 2),
                "negative_count": row.negative_count,
                "sample_description": (
                    row.sample_description[:200] if row.sample_description else None
                ),
            }
            issues.append(issue)

            if most_common is None:
                most_common = issue

        # Calculate percentages
        for issue in issues:
            issue["percentage"] = (
                round(issue["volume"] / total_tickets * 100, 1) if total_tickets > 0 else 0
            )

        return {
            "window_days": payload.window_days,
            "total_tickets": total_tickets,
            "issues": issues,
            "most_common_issue": most_common,
            "has_dominant_issue": most_common and most_common.get("percentage", 0) > 30,
        }


# =============================================================================
# GET COMPLAINT TRENDS TOOL
# =============================================================================


class ComplaintTrendsPayload(BaseModel):
    """Compare complaint volume between periods."""

    current_days: int = Field(default=1, ge=1, le=7)
    comparison_days: int = Field(default=7, ge=1, le=30)


class GetComplaintTrendsTool(BaseTool):
    """
    Compare complaint volume between current and previous period.
    Answers: "Did customer complaints increase yesterday?"
    """

    name = "get_complaint_trends"

    def request_model(self) -> type[BaseModel]:
        return ComplaintTrendsPayload

    async def run(self, session, payload: ComplaintTrendsPayload) -> dict[str, Any]:
        # Current period complaints
        current_stmt = text(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END) AS negative_count,
                AVG(sentiment) AS avg_sentiment
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :current_days || ' days'
        """
        )
        current_result = await session.execute(current_stmt, {"current_days": payload.current_days})
        current = current_result.one()

        # Comparison period (daily average)
        comparison_stmt = text(
            """
            SELECT
                COUNT(*) / NULLIF(:comparison_days, 0)::float AS daily_avg_total,
                SUM(CASE WHEN sentiment < 0.4 THEN 1 ELSE 0 END) / NULLIF(:comparison_days, 0)::float AS daily_avg_negative,
                AVG(sentiment) AS avg_sentiment
            FROM support_tickets
            WHERE created_at >= NOW() - INTERVAL :offset_days || ' days'
              AND created_at < NOW() - INTERVAL :current_days || ' days'
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

        current_total = current.total or 0
        expected_total = float(comparison.daily_avg_total or 0) * payload.current_days
        current_negative = current.negative_count or 0
        expected_negative = float(comparison.daily_avg_negative or 0) * payload.current_days

        # Calculate changes
        total_change = (
            ((current_total - expected_total) / expected_total * 100) if expected_total > 0 else 0
        )
        negative_change = (
            ((current_negative - expected_negative) / expected_negative * 100)
            if expected_negative > 0
            else 0
        )

        return {
            "current_period_days": payload.current_days,
            "comparison_period_days": payload.comparison_days,
            "current": {
                "total_tickets": current_total,
                "negative_tickets": current_negative,
                "avg_sentiment": round(float(current.avg_sentiment or 0), 2),
            },
            "expected": {
                "total_tickets": round(expected_total, 1),
                "negative_tickets": round(expected_negative, 1),
            },
            "change": {
                "total_pct": round(total_change, 1),
                "negative_pct": round(negative_change, 1),
            },
            "complaints_increased": total_change > 20 or negative_change > 20,
            "is_significant": abs(total_change) > 20 or abs(negative_change) > 30,
        }
