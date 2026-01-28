from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class GetSupportSentimentRequest(BaseModel):
    window_days: int = 7
    product_id: int | None = None


class SentimentStats(BaseModel):
    avg_sentiment: float
    negative_ratio: float
    ticket_volume: int


class GetSupportSentimentResponse(BaseModel):
    sentiment: SentimentStats


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
    trends: list[TicketTrend]
    alerts: list[str] | None = None


# =============================================================================
# HITL ACTION REQUEST/RESPONSE MODELS
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


# New request/response models for additional support tools
class GetCommonIssuesRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=30)
    min_count: int = Field(default=2, ge=1)
    limit: int = Field(default=10, ge=1, le=50)


class CommonIssue(BaseModel):
    issue_category: str
    ticket_count: int
    avg_sentiment: float
    negative_ratio: float
    sample_descriptions: list[str] = Field(default_factory=list)
    affected_products: list[str] = Field(default_factory=list)


class GetCommonIssuesResponse(BaseModel):
    window_days: int = 7
    common_issues: list[CommonIssue] = Field(default_factory=list)
    most_common_issue: str | None = None
    total_tickets_analyzed: int = 0
    has_critical_issues: bool = False


class GetComplaintTrendsRequest(BaseModel):
    current_days: int = Field(default=1, ge=1, le=30)
    previous_days: int = Field(default=7, ge=1, le=30)
    issue_category: str | None = None


class ComplaintCategoryTrend(BaseModel):
    issue_category: str
    current_count: int
    previous_avg: float
    change_pct: float
    trend: str


class GetComplaintTrendsResponse(BaseModel):
    current_days: int = 1
    previous_days: int = 7
    current_total: int = 0
    previous_avg: float = 0
    overall_change_pct: float = 0
    complaint_increased: bool = False
    category_trends: list[ComplaintCategoryTrend] = Field(default_factory=list)
    most_increased_category: str | None = None
    current_avg_sentiment: float | None = None
    previous_avg_sentiment: float | None = None


class SupportToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_support_sentiment(
        self, payload: GetSupportSentimentRequest
    ) -> GetSupportSentimentResponse:
        result = await self._client.invoke("get_support_sentiment", payload.model_dump())
        try:
            return GetSupportSentimentResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_support_sentiment: {exc}") from exc

    async def get_ticket_trends(self, payload: GetTicketTrendsRequest) -> GetTicketTrendsResponse:
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

    async def get_common_issues(self, payload: GetCommonIssuesRequest) -> GetCommonIssuesResponse:
        """Get most common customer issues."""
        result = await self._client.invoke("get_common_issues", payload.model_dump())
        try:
            return GetCommonIssuesResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_common_issues: {exc}") from exc

    async def get_complaint_trends(
        self, payload: GetComplaintTrendsRequest
    ) -> GetComplaintTrendsResponse:
        """Compare complaint volume between periods."""
        result = await self._client.invoke("get_complaint_trends", payload.model_dump())
        try:
            return GetComplaintTrendsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_complaint_trends: {exc}") from exc
