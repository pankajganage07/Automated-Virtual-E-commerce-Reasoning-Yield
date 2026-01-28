from __future__ import annotations

from typing import List, Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


class CampaignSpendPayload(BaseModel):
    campaign_ids: List[int] | None = None
    window_days: int = Field(default=7, ge=1, le=90)


class GetCampaignSpendTool(BaseTool):
    name = "get_campaign_spend"

    def request_model(self) -> type[BaseModel]:
        return CampaignSpendPayload

    async def run(self, session, payload: CampaignSpendPayload) -> dict[str, Any]:
        statement = """
            SELECT id, name, budget, spend, clicks, conversions, status
            FROM campaigns
            /* extend with time window logic if you track spend history */
            WHERE (:campaign_ids IS NULL OR id = ANY(:campaign_ids))
        """
        result = await session.execute(
            text(statement),
            {"campaign_ids": payload.campaign_ids},
        )
        rows = [dict(row._mapping) for row in result]
        totals = {
            "budget": float(sum(r["budget"] for r in rows)),
            "spend": float(sum(r["spend"] for r in rows)),
            "clicks": sum(r["clicks"] for r in rows),
            "conversions": sum(r["conversions"] for r in rows),
        }
        return {"summary": totals, "campaigns": rows}


# =============================================================================
# CALCULATE ROAS TOOL
# =============================================================================


class CalculateROASPayload(BaseModel):
    campaign_id: int | None = Field(None, description="Specific campaign ID (optional)")
    window_days: int = Field(default=7, ge=1, le=90, description="Analysis window in days")


class CalculateROASTool(BaseTool):
    """
    Calculate Return on Ad Spend (ROAS) for campaigns.

    ROAS = Revenue Generated / Ad Spend
    """

    name = "calculate_roas"

    def request_model(self) -> type[BaseModel]:
        return CalculateROASPayload

    async def run(self, session, payload: CalculateROASPayload) -> dict[str, Any]:
        # Get campaign spend data
        campaign_stmt = text(
            """
            SELECT id, name, budget, spend, clicks, conversions, status
            FROM campaigns
            WHERE (:campaign_id IS NULL OR id = :campaign_id)
        """
        )
        campaign_result = await session.execute(campaign_stmt, {"campaign_id": payload.campaign_id})
        campaigns = [dict(row._mapping) for row in campaign_result]

        if not campaigns:
            return {
                "error": (
                    f"Campaign {payload.campaign_id} not found"
                    if payload.campaign_id
                    else "No campaigns found"
                ),
                "roas_data": [],
            }

        # For each campaign, estimate revenue from conversions
        # In a real system, you'd track this more precisely
        # Here we use a simple heuristic: avg order value * conversions
        avg_order_stmt = text(
            """
            SELECT AVG(revenue) AS avg_order_value
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :window_days || ' days'
        """
        )
        avg_result = await session.execute(avg_order_stmt, {"window_days": payload.window_days})
        avg_row = avg_result.one()
        avg_order_value = float(avg_row.avg_order_value or 50.0)  # Default $50 if no data

        roas_data = []
        for campaign in campaigns:
            spend = float(campaign["spend"])
            conversions = campaign["conversions"]
            estimated_revenue = conversions * avg_order_value

            if spend > 0:
                roas = estimated_revenue / spend
            else:
                roas = 0.0

            # Determine performance tier
            if roas >= 4.0:
                performance = "excellent"
            elif roas >= 2.0:
                performance = "good"
            elif roas >= 1.0:
                performance = "break_even"
            else:
                performance = "poor"

            # Cost per conversion
            cpc = spend / conversions if conversions > 0 else None

            roas_data.append(
                {
                    "campaign_id": campaign["id"],
                    "campaign_name": campaign["name"],
                    "status": campaign["status"],
                    "spend": spend,
                    "conversions": conversions,
                    "estimated_revenue": round(estimated_revenue, 2),
                    "roas": round(roas, 2),
                    "performance": performance,
                    "cost_per_conversion": round(cpc, 2) if cpc else None,
                    "clicks": campaign["clicks"],
                    "conversion_rate": (
                        round(conversions / campaign["clicks"] * 100, 2)
                        if campaign["clicks"] > 0
                        else 0
                    ),
                }
            )

        # Calculate aggregate stats
        total_spend = sum(r["spend"] for r in roas_data)
        total_revenue = sum(r["estimated_revenue"] for r in roas_data)
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0

        return {
            "window_days": payload.window_days,
            "avg_order_value_used": round(avg_order_value, 2),
            "overall_roas": round(overall_roas, 2),
            "total_spend": round(total_spend, 2),
            "total_estimated_revenue": round(total_revenue, 2),
            "campaigns": roas_data,
        }


# =============================================================================
# GET UNDERPERFORMING CAMPAIGNS TOOL
# =============================================================================


class UnderperformingCampaignsPayload(BaseModel):
    """Find campaigns that are paused or underperforming."""

    include_paused: bool = Field(default=True, description="Include paused campaigns")
    roas_threshold: float = Field(default=1.0, description="ROAS below this is underperforming")


class GetUnderperformingCampaignsTool(BaseTool):
    """
    Find campaigns that are paused, have zero conversions, or poor ROAS.
    Answers: "Were any campaigns paused or underperforming?"
    """

    name = "get_underperforming_campaigns"

    def request_model(self) -> type[BaseModel]:
        return UnderperformingCampaignsPayload

    async def run(self, session, payload: UnderperformingCampaignsPayload) -> dict[str, Any]:
        # Get all campaigns with their performance
        stmt = text(
            """
            SELECT 
                id, name, budget, spend, clicks, conversions, status,
                CASE 
                    WHEN status = 'paused' THEN 'paused'
                    WHEN spend > 0 AND conversions = 0 THEN 'zero_conversions'
                    WHEN spend > budget * 0.8 AND conversions < 5 THEN 'low_efficiency'
                    ELSE 'ok'
                END AS performance_status
            FROM campaigns
            ORDER BY spend DESC
        """
        )
        result = await session.execute(stmt)

        # Get avg order value for ROAS calculation
        aov_stmt = text(
            """
            SELECT COALESCE(AVG(revenue), 50.0) AS avg_order_value
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL '7 days'
        """
        )
        aov_result = await session.execute(aov_stmt)
        avg_order_value = float(aov_result.one().avg_order_value)

        campaigns = []
        paused = []
        underperforming = []
        total_wasted_spend = 0

        for row in result:
            spend = float(row.spend)
            conversions = row.conversions
            estimated_revenue = conversions * avg_order_value
            roas = estimated_revenue / spend if spend > 0 else 0

            campaign = {
                "campaign_id": row.id,
                "name": row.name,
                "status": row.status,
                "budget": float(row.budget),
                "spend": spend,
                "clicks": row.clicks,
                "conversions": conversions,
                "roas": round(roas, 2),
                "performance_status": row.performance_status,
            }
            campaigns.append(campaign)

            if row.status == "paused" and payload.include_paused:
                paused.append(campaign)
            elif (
                row.performance_status in ("zero_conversions", "low_efficiency")
                or roas < payload.roas_threshold
            ):
                underperforming.append(campaign)
                if roas < payload.roas_threshold:
                    total_wasted_spend += spend * (1 - roas)

        return {
            "all_campaigns": campaigns,
            "paused_campaigns": paused,
            "underperforming_campaigns": underperforming,
            "has_issues": len(paused) > 0 or len(underperforming) > 0,
            "total_paused": len(paused),
            "total_underperforming": len(underperforming),
            "estimated_wasted_spend": round(total_wasted_spend, 2),
        }


# =============================================================================
# COMPARE CAMPAIGN PERFORMANCE TOOL
# =============================================================================


class CompareCampaignPerformancePayload(BaseModel):
    """Compare campaign performance between two periods."""

    current_days: int = Field(default=7, ge=1, le=30)
    comparison_days: int = Field(default=7, ge=1, le=60)


class CompareCampaignPerformanceTool(BaseTool):
    """
    Compare campaign performance between current period and previous period.
    Note: This assumes campaigns table has historical data or uses a simplified comparison.
    """

    name = "compare_campaign_performance"

    def request_model(self) -> type[BaseModel]:
        return CompareCampaignPerformancePayload

    async def run(self, session, payload: CompareCampaignPerformancePayload) -> dict[str, Any]:
        # Since campaigns table is cumulative, we'll compare current metrics
        # In a real system, you'd have campaign_metrics_history table
        stmt = text(
            """
            SELECT 
                id, name, budget, spend, clicks, conversions, status
            FROM campaigns
        """
        )
        result = await session.execute(stmt)

        # Get conversion trend from orders
        orders_stmt = text(
            """
            SELECT
                'current' AS period,
                COUNT(*) AS order_count,
                COALESCE(SUM(revenue), 0) AS revenue
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :current_days || ' days'
            UNION ALL
            SELECT
                'previous' AS period,
                COUNT(*) AS order_count,
                COALESCE(SUM(revenue), 0) AS revenue
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :offset_days || ' days'
              AND timestamp < NOW() - INTERVAL :current_days || ' days'
        """
        )
        orders_result = await session.execute(
            orders_stmt,
            {
                "current_days": payload.current_days,
                "offset_days": payload.current_days + payload.comparison_days,
            },
        )
        periods = {
            row.period: {"orders": row.order_count, "revenue": float(row.revenue)}
            for row in orders_result
        }

        current_orders = periods.get("current", {}).get("orders", 0)
        prev_orders = periods.get("previous", {}).get("orders", 0)
        current_revenue = periods.get("current", {}).get("revenue", 0)
        prev_revenue = periods.get("previous", {}).get("revenue", 0)

        # Calculate changes
        orders_change = (
            ((current_orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0
        )
        revenue_change = (
            ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        )

        campaigns = [dict(row._mapping) for row in result]
        total_spend = sum(float(c["spend"]) for c in campaigns)
        total_conversions = sum(c["conversions"] for c in campaigns)

        return {
            "current_period_days": payload.current_days,
            "comparison_period_days": payload.comparison_days,
            "campaigns": campaigns,
            "total_spend": round(total_spend, 2),
            "total_conversions": total_conversions,
            "orders_trend": {
                "current": current_orders,
                "previous": prev_orders,
                "change_pct": round(orders_change, 1),
            },
            "revenue_trend": {
                "current": round(current_revenue, 2),
                "previous": round(prev_revenue, 2),
                "change_pct": round(revenue_change, 1),
            },
            "performance_dropped": orders_change < -10 or revenue_change < -10,
        }
