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
