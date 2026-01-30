"""
Marketing tools for LangGraph agents.

Slimmed toolset (2 core tools):
1. get_campaign_spend - Get spend and conversion metrics
2. calculate_roas - Calculate Return on Ad Spend

Complex queries should route to DataAnalystAgent with HITL.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


# =============================================================================
# GET CAMPAIGN SPEND (Core Tool #1)
# =============================================================================


class GetCampaignSpendRequest(BaseModel):
    campaign_ids: list[int] | None = Field(default=None, description="Optional campaign IDs filter")
    status: str | None = Field(default=None, description="Filter by status: active, paused")


class CampaignInfo(BaseModel):
    campaign_id: int
    name: str
    budget: float
    spend: float
    clicks: int
    conversions: int
    status: str
    budget_utilization_pct: float = 0


class GetCampaignSpendResponse(BaseModel):
    summary: dict[str, float] = Field(default_factory=dict)
    campaigns: list[CampaignInfo] = Field(default_factory=list)
    campaign_count: int = 0


# =============================================================================
# CALCULATE ROAS (Core Tool #2)
# =============================================================================


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

    window_days: int = 7
    avg_order_value_used: float = 0
    overall_roas: float = 0
    total_spend: float = 0
    total_estimated_revenue: float = 0
    campaigns: list[CampaignROASInfo] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# ACTION TOOLS (for HITL execution)
# =============================================================================


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


# =============================================================================
# MARKETING TOOLSET
# =============================================================================


class MarketingToolset:
    """
    Marketing tools (2 core tools).

    For complex queries (underperforming campaigns, campaign comparison),
    the agent should indicate it cannot handle the query, and the supervisor
    will route to the DataAnalystAgent.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_campaign_spend(
        self, payload: GetCampaignSpendRequest
    ) -> GetCampaignSpendResponse:
        """Get spend and conversion metrics for campaigns."""
        result = await self._client.invoke("get_campaign_spend", payload.model_dump())
        try:
            return GetCampaignSpendResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_campaign_spend: {exc}") from exc

    async def calculate_roas(self, payload: CalculateROASRequest) -> CalculateROASResponse:
        """Calculate ROAS for campaigns."""
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
