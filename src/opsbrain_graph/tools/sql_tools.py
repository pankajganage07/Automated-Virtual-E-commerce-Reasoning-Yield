"""
SQL and Sales tools for LangGraph agents.

Slimmed toolsets:
- SalesToolset: get_sales_summary, get_top_products (2 core tools)
- Complex queries route to DataAnalystAgent with HITL
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .mcp_client import MCPClient
from .exceptions import MCPError


# =============================================================================
# SQL TOOL (for approved custom queries via HITL)
# =============================================================================


class ExecuteSQLRequest(BaseModel):
    statement: str
    params: dict[str, Any] | None = None
    fetch: Literal["all", "one", "value"] = "all"


class ExecuteSQLResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    rowcount: int = 0
    columns: list[str] = Field(default_factory=list)


class SQLToolset:
    """Raw SQL execution toolset - used by action_executor for approved queries."""

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def execute(self, payload: ExecuteSQLRequest) -> ExecuteSQLResponse:
        result = await self._client.invoke("execute_sql_query", payload.model_dump())
        try:
            return ExecuteSQLResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for execute_sql_query: {exc}") from exc


# =============================================================================
# SALES TOOLSET (2 core tools)
# =============================================================================


class GetSalesSummaryRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    group_by: str = Field(default="day")


class SalesSummaryResponse(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    trend: list[dict[str, Any]] = Field(default_factory=list)
    trend_analysis: str = Field(default="stable")


class GetTopProductsRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=5, ge=1, le=50)


class TopProduct(BaseModel):
    product_id: int
    name: str
    category: str | None = None
    units_sold: int
    revenue: float


class GetTopProductsResponse(BaseModel):
    products: list[TopProduct] = Field(default_factory=list)
    window_days: int = 7
    total_top_products_revenue: float = 0


class SalesToolset:
    """
    Sales-specific tools (2 core tools).

    For complex queries (compare periods, regional, channel, product contribution),
    the agent should indicate it cannot handle the query, and the supervisor
    will route to the DataAnalystAgent.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_sales_summary(self, payload: GetSalesSummaryRequest) -> SalesSummaryResponse:
        """Get aggregated sales metrics with trend analysis."""
        result = await self._client.invoke("get_sales_summary", payload.model_dump())
        try:
            return SalesSummaryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_sales_summary: {exc}") from exc

    async def get_top_products(self, payload: GetTopProductsRequest) -> GetTopProductsResponse:
        """Get best selling products by revenue."""
        result = await self._client.invoke("get_top_products", payload.model_dump())
        try:
            return GetTopProductsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_top_products: {exc}") from exc
