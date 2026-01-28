from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class GetCampaignSpendRequest(BaseModel):
    campaign_ids: list[int] | None = None
    window_days: int = 7


class CampaignInfo(BaseModel):
    id: int
    name: str
    budget: float
    spend: float
    clicks: int
    conversions: int
    status: str


class GetCampaignSpendResponse(BaseModel):
    summary: dict[str, float]
    campaigns: list[CampaignInfo]


class CalculateROASRequest(BaseModel):
    """Request to calculate ROAS for campaigns."""

    campaign_id: int | None = Field(
        None, description="Specific campaign ID (optional, all if None)"
    )
    window_days: int = Field(default=7, ge=1, le=90, description="Analysis window in days")


class CampaignROASInfo(BaseModel):
    """ROAS data for a single campaign."""

    campaign_id: int
    campaign_name: str
    status: str
    spend: float
    conversions: int
    estimated_revenue: float
    roas: float
    performance: str  # excellent, good, break_even, poor
    cost_per_conversion: float | None = None
    clicks: int
    conversion_rate: float


class CalculateROASResponse(BaseModel):
    """Response from ROAS calculation."""

    window_days: int | None = None
    avg_order_value_used: float | None = None
    overall_roas: float | None = None
    total_spend: float | None = None
    total_estimated_revenue: float | None = None
    campaigns: list[CampaignROASInfo] | None = None
    # Legacy fields for backward compatibility
    campaign_id: int | None = None
    roas: float | None = None
    notes: str | None = None
    error: str | None = None


class PauseCampaignRequest(BaseModel):
    """Request to pause a campaign (HITL action)."""

    campaign_id: int = Field(..., description="Campaign ID to pause")
    reason: str | None = Field(None, description="Reason for pausing")


class PauseCampaignResponse(BaseModel):
    """Response from pause campaign operation."""

    success: bool
    campaign_id: int | None = None
    campaign_name: str | None = None
    old_status: str | None = None
    new_status: str | None = None
    reason: str | None = None
    error: str | None = None


class ResumeCampaignRequest(BaseModel):
    """Request to resume a paused campaign (HITL action)."""

    campaign_id: int = Field(..., description="Campaign ID to resume")
    reason: str | None = Field(None, description="Reason for resuming")


class ResumeCampaignResponse(BaseModel):
    """Response from resume campaign operation."""

    success: bool
    campaign_id: int | None = None
    campaign_name: str | None = None
    old_status: str | None = None
    new_status: str | None = None
    reason: str | None = None
    error: str | None = None


class AdjustBudgetRequest(BaseModel):
    """Request to adjust campaign budget (HITL action)."""

    campaign_id: int = Field(..., description="Campaign ID to adjust")
    new_budget: float = Field(..., gt=0, description="New budget amount")
    reason: str | None = Field(None, description="Reason for adjustment")


class AdjustBudgetResponse(BaseModel):
    """Response from budget adjustment operation."""

    success: bool
    campaign_id: int | None = None
    campaign_name: str | None = None
    old_budget: float | None = None
    new_budget: float | None = None
    reason: str | None = None
    error: str | None = None


# New request/response models for additional marketing tools
class GetUnderperformingCampaignsRequest(BaseModel):
    min_spend: float = Field(default=0, ge=0)
    include_paused: bool = Field(default=True)


class UnderperformingCampaign(BaseModel):
    campaign_id: int
    name: str
    status: str
    budget: float
    spend: float
    clicks: int
    conversions: int
    roas: float
    issue: str


class GetUnderperformingCampaignsResponse(BaseModel):
    underperforming_campaigns: list[UnderperformingCampaign] = Field(default_factory=list)
    total_count: int = 0
    paused_count: int = 0
    zero_conversion_count: int = 0
    poor_roas_count: int = 0
    has_issues: bool = False


class CompareCampaignPerformanceRequest(BaseModel):
    current_days: int = Field(default=1, ge=1, le=30)
    previous_days: int = Field(default=7, ge=1, le=30)
    campaign_ids: list[int] | None = None


class CampaignPerformanceComparison(BaseModel):
    campaign_id: int
    name: str
    current_spend: float
    previous_spend: float
    current_conversions: int
    previous_conversions: int
    current_roas: float
    previous_roas: float
    spend_change_pct: float
    conversion_change_pct: float
    roas_change_pct: float
    trend: str


class CompareCampaignPerformanceResponse(BaseModel):
    current_days: int = 1
    previous_days: int = 7
    campaigns: list[CampaignPerformanceComparison] = Field(default_factory=list)
    total_current_spend: float = 0
    total_previous_spend: float = 0
    overall_spend_change_pct: float = 0
    total_current_conversions: int = 0
    total_previous_conversions: int = 0
    overall_conversion_change_pct: float = 0
    declining_campaigns_count: int = 0


class MarketingToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_campaign_spend(
        self, payload: GetCampaignSpendRequest
    ) -> GetCampaignSpendResponse:
        result = await self._client.invoke("get_campaign_spend", payload.model_dump())
        try:
            return GetCampaignSpendResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_campaign_spend: {exc}") from exc

    async def calculate_roas(self, payload: CalculateROASRequest) -> CalculateROASResponse:
        result = await self._client.invoke("calculate_roas", payload.model_dump())
        try:
            return CalculateROASResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for calculate_roas: {exc}") from exc

    async def pause_campaign(self, payload: PauseCampaignRequest) -> PauseCampaignResponse:
        """Pause a campaign (HITL action - calls update_campaign_status MCP tool)."""
        mcp_payload = {
            "campaign_id": payload.campaign_id,
            "status": "paused",
            "reason": payload.reason,
        }
        result = await self._client.invoke("update_campaign_status", mcp_payload)
        try:
            return PauseCampaignResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for pause_campaign: {exc}") from exc

    async def resume_campaign(self, payload: ResumeCampaignRequest) -> ResumeCampaignResponse:
        """Resume a paused campaign (HITL action - calls update_campaign_status MCP tool)."""
        mcp_payload = {
            "campaign_id": payload.campaign_id,
            "status": "active",
            "reason": payload.reason,
        }
        result = await self._client.invoke("update_campaign_status", mcp_payload)
        try:
            return ResumeCampaignResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for resume_campaign: {exc}") from exc

    async def adjust_budget(self, payload: AdjustBudgetRequest) -> AdjustBudgetResponse:
        """Adjust campaign budget (HITL action - calls update_campaign_budget MCP tool)."""
        result = await self._client.invoke("update_campaign_budget", payload.model_dump())
        try:
            return AdjustBudgetResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for adjust_budget: {exc}") from exc

    async def get_underperforming_campaigns(
        self, payload: GetUnderperformingCampaignsRequest
    ) -> GetUnderperformingCampaignsResponse:
        """Get underperforming or paused campaigns."""
        result = await self._client.invoke("get_underperforming_campaigns", payload.model_dump())
        try:
            return GetUnderperformingCampaignsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_underperforming_campaigns: {exc}") from exc

    async def compare_campaign_performance(
        self, payload: CompareCampaignPerformanceRequest
    ) -> CompareCampaignPerformanceResponse:
        """Compare campaign performance between periods."""
        result = await self._client.invoke("compare_campaign_performance", payload.model_dump())
        try:
            return CompareCampaignPerformanceResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for compare_campaign_performance: {exc}") from exc
