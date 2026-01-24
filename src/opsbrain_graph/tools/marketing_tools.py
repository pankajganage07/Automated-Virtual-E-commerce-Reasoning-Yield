from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class GetAdSpendRequest(BaseModel):
    campaign_ids: list[int] | None = None
    window_days: int = 7


class CampaignSpend(BaseModel):
    campaign_id: int
    spend: float
    clicks: int
    conversions: int
    status: str


class GetAdSpendResponse(BaseModel):
    summary: dict[str, float]
    campaigns: list[CampaignSpend]


class CalculateROASRequest(BaseModel):
    campaign_id: int
    revenue: float
    spend: float


class CalculateROASResponse(BaseModel):
    campaign_id: int
    roas: float
    notes: str | None = None


class PauseCampaignRequest(BaseModel):
    campaign_id: int
    reason: str


class PauseCampaignResponse(BaseModel):
    pending_action_id: str
    message: str


class MarketingToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_ad_spend(self, payload: GetAdSpendRequest) -> GetAdSpendResponse:
        result = await self._client.invoke("get_ad_spend", payload.model_dump())
        try:
            return GetAdSpendResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_ad_spend: {exc}") from exc

    async def calculate_roas(self, payload: CalculateROASRequest) -> CalculateROASResponse:
        result = await self._client.invoke("calculate_roas", payload.model_dump())
        try:
            return CalculateROASResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for calculate_roas: {exc}") from exc

    async def pause_campaign(self, payload: PauseCampaignRequest) -> PauseCampaignResponse:
        result = await self._client.invoke("pause_campaign", payload.model_dump())
        try:
            return PauseCampaignResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for pause_campaign: {exc}") from exc
