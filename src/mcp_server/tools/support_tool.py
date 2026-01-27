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
