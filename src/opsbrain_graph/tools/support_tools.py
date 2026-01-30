"""
Support tools for LangGraph agents.

Slimmed toolset (2 core tools):
1. get_support_sentiment - Get sentiment analysis and ticket volume
2. get_ticket_trends - Analyze trends by category/product/day

Complex queries should route to DataAnalystAgent with HITL.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


# =============================================================================
# GET SUPPORT SENTIMENT (Core Tool #1)
# =============================================================================


class GetSupportSentimentRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90, description="Analysis window in days")
    product_id: int | None = Field(None, description="Optional product filter")


class SentimentStats(BaseModel):
    avg_sentiment: float
    negative_ratio: float
    ticket_volume: int


class GetSupportSentimentResponse(BaseModel):
    sentiment: SentimentStats


# =============================================================================
# GET TICKET TRENDS (Core Tool #2)
# =============================================================================


class GetTicketTrendsRequest(BaseModel):
    """Request for ticket trend analysis."""

    window_days: int = Field(default=14, ge=1, le=90, description="Analysis window in days")
    group_by: str = Field(
        default="issue_category", description="Group by: issue_category, product, day"
    )
    product_id: int | None = Field(None, description="Optional product filter")


class TicketTrend(BaseModel):
    """Trend data for a single category/group."""

    key: str
    volume: int
    previous_volume: int | None = None
    change_pct: float
    trend: str | None = None  # increasing, decreasing, stable
    avg_sentiment: float | None = None
    negative_count: int | None = None


class GetTicketTrendsResponse(BaseModel):
    """Response from ticket trends analysis."""

    window_days: int | None = None
    group_by: str | None = None
    total_volume: int | None = None
    trends: list[TicketTrend] = Field(default_factory=list)
    alerts: list[str] | None = None


# =============================================================================
# ACTION TOOLS (for HITL execution)
# =============================================================================


class EscalateTicketRequest(BaseModel):
    """Request to escalate a support ticket (HITL action)."""

    ticket_id: int = Field(..., description="Ticket ID to escalate")
    priority: str = Field(default="high", description="New priority level")
    reason: str | None = Field(None, description="Reason for escalation")


class EscalateTicketResponse(BaseModel):
    """Response from ticket escalation."""

    success: bool
    ticket_id: int | None = None
    issue_category: str | None = None
    new_priority: str | None = None
    reason: str | None = None
    note: str | None = None
    error: str | None = None


class CloseTicketRequest(BaseModel):
    """Request to close a support ticket (HITL action)."""

    ticket_id: int = Field(..., description="Ticket ID to close")
    resolution: str | None = Field(None, description="Resolution summary")


class CloseTicketResponse(BaseModel):
    """Response from ticket closure."""

    success: bool
    ticket_id: int | None = None
    issue_category: str | None = None
    resolution: str | None = None
    note: str | None = None
    error: str | None = None


class PrioritizeTicketRequest(BaseModel):
    """Request to set priority for a support ticket (HITL action)."""

    ticket_id: int = Field(..., description="Ticket ID to prioritize")
    priority: str = Field(
        default="medium", description="Priority level: low, medium, high, critical"
    )


class PrioritizeTicketResponse(BaseModel):
    """Response from priority update."""

    success: bool
    ticket_id: int | None = None
    issue_category: str | None = None
    priority: str | None = None
    note: str | None = None
    error: str | None = None


# =============================================================================
# SUPPORT TOOLSET
# =============================================================================


class SupportToolset:
    """
    Support tools (2 core tools).

    For complex queries (common issues analysis, complaint trend comparison),
    the agent should indicate it cannot handle the query, and the supervisor
    will route to the DataAnalystAgent.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    # =========================================================================
    # CORE READ OPERATIONS
    # =========================================================================

    async def get_support_sentiment(
        self, payload: GetSupportSentimentRequest
    ) -> GetSupportSentimentResponse:
        """Get sentiment analysis and ticket volume metrics."""
        result = await self._client.invoke("get_support_sentiment", payload.model_dump())
        try:
            return GetSupportSentimentResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_support_sentiment: {exc}") from exc

    async def get_ticket_trends(self, payload: GetTicketTrendsRequest) -> GetTicketTrendsResponse:
        """Analyze ticket trends by category, product, or day."""
        result = await self._client.invoke("get_ticket_trends", payload.model_dump())
        try:
            return GetTicketTrendsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_ticket_trends: {exc}") from exc

    # =========================================================================
    # HITL ACTION OPERATIONS
    # =========================================================================

    async def escalate_ticket(self, payload: EscalateTicketRequest) -> EscalateTicketResponse:
        """Escalate a support ticket to higher priority (HITL action)."""
        result = await self._client.invoke("escalate_ticket", payload.model_dump())
        try:
            return EscalateTicketResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for escalate_ticket: {exc}") from exc

    async def close_ticket(self, payload: CloseTicketRequest) -> CloseTicketResponse:
        """Close a support ticket (HITL action)."""
        result = await self._client.invoke("close_ticket", payload.model_dump())
        try:
            return CloseTicketResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for close_ticket: {exc}") from exc

    async def prioritize_ticket(self, payload: PrioritizeTicketRequest) -> PrioritizeTicketResponse:
        """Set priority level for a support ticket (HITL action)."""
        result = await self._client.invoke("prioritize_ticket", payload.model_dump())
        try:
            return PrioritizeTicketResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for prioritize_ticket: {exc}") from exc
