from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .mcp_client import MCPClient
from .exceptions import MCPError


class ExecuteSQLRequest(BaseModel):
    statement: str
    params: dict[str, Any] | None = None
    fetch: Literal["all", "one", "value"] = "all"


class ExecuteSQLResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    rowcount: int = 0
    columns: list[str] = Field(default_factory=list)


class SQLToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def execute(self, payload: ExecuteSQLRequest) -> ExecuteSQLResponse:
        result = await self._client.invoke("execute_sql_query", payload.model_dump())
        try:
            return ExecuteSQLResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for execute_sql_query: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Sales-specific tools
# ─────────────────────────────────────────────────────────────────────────────


class GetTopProductsRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=5, ge=1, le=50)


class TopProduct(BaseModel):
    product_id: int
    name: str
    units_sold: int
    revenue: float


class GetTopProductsResponse(BaseModel):
    products: list[TopProduct] = Field(default_factory=list)


class GetSalesSummaryRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    group_by: str = Field(default="day")


class SalesSummaryResponse(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    trend: list[dict[str, Any]] = Field(default_factory=list)


# New request/response models for additional sales tools
class CompareSalesPeriodsRequest(BaseModel):
    current_days: int = Field(default=1, ge=1, le=30)
    previous_days: int = Field(default=7, ge=1, le=90)


class CompareSalesPeriodsResponse(BaseModel):
    current_days: int = 1
    previous_days: int = 7
    current_revenue: float = 0
    previous_revenue: float = 0
    current_orders: int = 0
    previous_orders: int = 0
    revenue_change_pct: float = 0
    order_change_pct: float = 0
    avg_order_value_change_pct: float = 0
    trend: str = "stable"


class GetRegionalSalesRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    compare_to_avg: bool = Field(default=True)


class RegionalSales(BaseModel):
    region: str
    revenue: float
    orders: int
    avg_revenue: float = 0
    change_pct: float = 0


class GetRegionalSalesResponse(BaseModel):
    window_days: int = 7
    total_revenue: float = 0
    regions: list[RegionalSales] = Field(default_factory=list)
    underperforming_regions: list[RegionalSales] = Field(default_factory=list)
    top_region: str | None = None
    worst_region: str | None = None


class GetChannelPerformanceRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)


class ChannelSales(BaseModel):
    channel: str
    revenue: float
    orders: int
    avg_order_value: float = 0
    revenue_share: float = 0


class GetChannelPerformanceResponse(BaseModel):
    window_days: int = 7
    total_revenue: float = 0
    channels: list[ChannelSales] = Field(default_factory=list)
    top_channel: str | None = None
    worst_channel: str | None = None


class GetProductContributionRequest(BaseModel):
    current_days: int = Field(default=1, ge=1, le=7)
    previous_days: int = Field(default=7, ge=1, le=30)
    limit: int = Field(default=10, ge=1, le=50)


class ProductContribution(BaseModel):
    product_id: int
    name: str
    current_revenue: float = 0
    previous_avg_revenue: float = 0
    change_pct: float = 0
    contribution_pct: float = 0


class GetProductContributionResponse(BaseModel):
    current_days: int = 1
    previous_days: int = 7
    overall_change_pct: float = 0
    products: list[ProductContribution] = Field(default_factory=list)
    biggest_gainers: list[ProductContribution] = Field(default_factory=list)
    biggest_losers: list[ProductContribution] = Field(default_factory=list)


class SalesToolset:
    """Sales-specific tools that wrap MCP server endpoints."""

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_top_products(self, payload: GetTopProductsRequest) -> GetTopProductsResponse:
        result = await self._client.invoke("get_top_products", payload.model_dump())
        try:
            return GetTopProductsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_top_products: {exc}") from exc

    async def get_sales_summary(self, payload: GetSalesSummaryRequest) -> SalesSummaryResponse:
        result = await self._client.invoke("get_sales_summary", payload.model_dump())
        try:
            return SalesSummaryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_sales_summary: {exc}") from exc

    async def compare_sales_periods(
        self, payload: CompareSalesPeriodsRequest
    ) -> CompareSalesPeriodsResponse:
        result = await self._client.invoke("compare_sales_periods", payload.model_dump())
        try:
            return CompareSalesPeriodsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for compare_sales_periods: {exc}") from exc

    async def get_regional_sales(
        self, payload: GetRegionalSalesRequest
    ) -> GetRegionalSalesResponse:
        result = await self._client.invoke("get_regional_sales", payload.model_dump())
        try:
            return GetRegionalSalesResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_regional_sales: {exc}") from exc

    async def get_channel_performance(
        self, payload: GetChannelPerformanceRequest
    ) -> GetChannelPerformanceResponse:
        result = await self._client.invoke("get_channel_performance", payload.model_dump())
        try:
            return GetChannelPerformanceResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_channel_performance: {exc}") from exc

    async def get_product_contribution(
        self, payload: GetProductContributionRequest
    ) -> GetProductContributionResponse:
        result = await self._client.invoke("get_product_contribution", payload.model_dump())
        try:
            return GetProductContributionResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_product_contribution: {exc}") from exc
