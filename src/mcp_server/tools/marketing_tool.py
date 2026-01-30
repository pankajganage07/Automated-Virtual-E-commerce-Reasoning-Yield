"""
MCP Tools for marketing domain.

Core tools (2 main):
1. get_campaign_spend - Get spend and conversion metrics for campaigns
2. calculate_roas - Calculate Return on Ad Spend

Complex queries (underperforming campaigns, campaign comparison)
should be routed to the Data Analyst agent with HITL approval.
"""

from __future__ import annotations

from typing import List, Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


# =============================================================================
# GET CAMPAIGN SPEND TOOL
# =============================================================================


class CampaignSpendPayload(BaseModel):
    campaign_ids: List[int] | None = Field(default=None, description="Optional campaign IDs filter")
    status: str | None = Field(default=None, description="Filter by status: active, paused")


class GetCampaignSpendTool(BaseTool):
    """Return spend, budget, and conversion metrics for campaigns."""

    name = "get_campaign_spend"

    def request_model(self) -> type[BaseModel]:
        return CampaignSpendPayload

    async def run(self, session, payload: CampaignSpendPayload) -> dict[str, Any]:
        # Build query dynamically to avoid NULL parameter type issues with asyncpg
        conditions = []
        params = {}

        if payload.campaign_ids is not None:
            conditions.append("id = ANY(:campaign_ids)")
            params["campaign_ids"] = payload.campaign_ids

        if payload.status is not None:
            conditions.append("status = :status")
            params["status"] = payload.status

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        statement = f"""
            SELECT 
                id, 
                name, 
                budget, 
                spend, 
                clicks, 
                conversions, 
                status,
                CASE 
                    WHEN budget > 0 THEN ROUND((spend / budget) * 100, 1)
                    ELSE 0 
                END AS budget_utilization_pct
            FROM campaigns
            {where_clause}
            ORDER BY spend DESC
        """
        result = await session.execute(text(statement), params)

        campaigns = []
        for row in result:
            campaigns.append(
                {
                    "campaign_id": row.id,
                    "name": row.name,
                    "budget": float(row.budget),
                    "spend": float(row.spend),
                    "clicks": row.clicks,
                    "conversions": row.conversions,
                    "status": row.status,
                    "budget_utilization_pct": float(row.budget_utilization_pct),
                }
            )

        totals = {
            "total_budget": sum(c["budget"] for c in campaigns),
            "total_spend": sum(c["spend"] for c in campaigns),
            "total_clicks": sum(c["clicks"] for c in campaigns),
            "total_conversions": sum(c["conversions"] for c in campaigns),
        }

        return {
            "summary": totals,
            "campaigns": campaigns,
            "campaign_count": len(campaigns),
        }


# =============================================================================
# CALCULATE ROAS TOOL
# =============================================================================


class CalculateROASPayload(BaseModel):
    campaign_id: int | None = Field(None, description="Specific campaign ID (optional)")
    window_days: int = Field(
        default=7, ge=1, le=90, description="Analysis window for avg order value"
    )


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
        campaigns = list(campaign_result)

        if not campaigns:
            return {
                "error": (
                    f"Campaign {payload.campaign_id} not found"
                    if payload.campaign_id
                    else "No campaigns found"
                ),
                "roas_data": [],
            }

        # Get average order value for revenue estimation
        avg_order_stmt = text(
            """
            SELECT COALESCE(AVG(revenue), 50.0) AS avg_order_value
            FROM orders
            WHERE timestamp >= NOW() - INTERVAL :window_days || ' days'
        """
        )
        avg_result = await session.execute(avg_order_stmt, {"window_days": payload.window_days})
        avg_row = avg_result.one()
        avg_order_value = float(avg_row.avg_order_value)

        roas_data = []
        for row in campaigns:
            spend = float(row.spend)
            conversions = row.conversions
            estimated_revenue = conversions * avg_order_value

            roas = (estimated_revenue / spend) if spend > 0 else 0.0

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
            cpc = (spend / conversions) if conversions > 0 else None

            roas_data.append(
                {
                    "campaign_id": row.id,
                    "campaign_name": row.name,
                    "status": row.status,
                    "spend": spend,
                    "conversions": conversions,
                    "estimated_revenue": round(estimated_revenue, 2),
                    "roas": round(roas, 2),
                    "performance": performance,
                    "cost_per_conversion": round(cpc, 2) if cpc else None,
                    "clicks": row.clicks,
                    "conversion_rate": (
                        round(conversions / row.clicks * 100, 2) if row.clicks > 0 else 0
                    ),
                }
            )

        # Calculate aggregate stats
        total_spend = sum(r["spend"] for r in roas_data)
        total_revenue = sum(r["estimated_revenue"] for r in roas_data)
        overall_roas = (total_revenue / total_spend) if total_spend > 0 else 0

        return {
            "window_days": payload.window_days,
            "avg_order_value_used": round(avg_order_value, 2),
            "overall_roas": round(overall_roas, 2),
            "total_spend": round(total_spend, 2),
            "total_estimated_revenue": round(total_revenue, 2),
            "campaigns": roas_data,
        }
