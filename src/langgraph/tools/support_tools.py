from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class AnalyzeSentimentRequest(BaseModel):
    window_days: int = 7
    product_id: int | None = None


class SentimentStats(BaseModel):
    avg_sentiment: float
    negative_ratio: float
    volume: int
    top_issues: list[str]


class AnalyzeSentimentResponse(BaseModel):
    sentiment: SentimentStats
    sample_tickets: list[dict]


class GetTicketTrendsRequest(BaseModel):
    window_days: int = 14
    group_by: str = "issue_category"


class TicketTrend(BaseModel):
    key: str
    volume: int
    change_pct: float


class GetTicketTrendsResponse(BaseModel):
    trends: list[TicketTrend]


class SupportToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def analyze_sentiment(self, payload: AnalyzeSentimentRequest) -> AnalyzeSentimentResponse:
        result = await self._client.invoke("analyze_sentiment", payload.model_dump())
        try:
            return AnalyzeSentimentResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for analyze_sentiment: {exc}") from exc

    async def get_ticket_trends(self, payload: GetTicketTrendsRequest) -> GetTicketTrendsResponse:
        result = await self._client.invoke("get_ticket_trends", payload.model_dump())
        try:
            return GetTicketTrendsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_ticket_trends: {exc}") from exc
