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
