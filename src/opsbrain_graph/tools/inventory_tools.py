from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class GetInventoryStatusRequest(BaseModel):
    product_ids: list[int] = Field(..., min_length=1)


class InventoryItem(BaseModel):
    id: int
    name: str
    stock_qty: int
    low_stock_threshold: int


class GetInventoryStatusResponse(BaseModel):
    items: list[InventoryItem]


class PredictStockOutRequest(BaseModel):
    product_id: int = Field(..., description="Product ID to analyze")
    lookahead_days: int = Field(default=7, ge=1, le=30, description="Days to look ahead")


class PredictStockOutResponse(BaseModel):
    """Response from stock-out prediction."""

    product_id: int
    product_name: str | None = None
    current_stock: int | None = None
    low_stock_threshold: int | None = None
    avg_daily_sales: float | None = None
    days_until_stockout: float | None = None
    projected_stockout_date: str | None = None
    within_lookahead_window: bool | None = None
    confidence: float | None = None
    risk_level: str | None = None  # critical, high, medium, low, unknown
    message: str | None = None
    error: str | None = None


class RestockItemRequest(BaseModel):
    """Request to restock a product (HITL action)."""

    product_id: int = Field(..., description="Product ID to restock")
    quantity: int = Field(..., gt=0, description="Quantity to add to stock")
    reason: str | None = Field(None, description="Reason for restocking")


class RestockItemResponse(BaseModel):
    """Response from restock operation."""

    success: bool
    product_id: int
    product_name: str | None = None
    old_quantity: int | None = None
    new_quantity: int | None = None
    change: int | None = None
    reason: str | None = None
    error: str | None = None


class UpdateInventoryRequest(BaseModel):
    """Request to update inventory (generic HITL action)."""

    product_id: int = Field(..., description="Product ID to update")
    quantity_change: int = Field(..., description="Amount to add (positive) or remove (negative)")
    reason: str | None = Field(None, description="Reason for adjustment")


class UpdateInventoryResponse(BaseModel):
    """Response from inventory update operation."""

    success: bool
    product_id: int | None = None
    product_name: str | None = None
    old_quantity: int | None = None
    new_quantity: int | None = None
    change: int | None = None
    reason: str | None = None
    error: str | None = None


# New request/response models for additional inventory tools
class GetLowStockProductsRequest(BaseModel):
    include_out_of_stock: bool = Field(default=True)
    limit: int = Field(default=20, ge=1, le=100)


class LowStockProduct(BaseModel):
    product_id: int
    name: str
    category: str | None = None
    stock_qty: int
    low_stock_threshold: int
    buffer: int
    status: str
    needs_restock: bool = True


class GetLowStockProductsResponse(BaseModel):
    low_stock_products: list[LowStockProduct] = Field(default_factory=list)
    total_count: int = 0
    out_of_stock_count: int = 0
    critical_count: int = 0
    has_critical: bool = False


class CheckTopSellersStockRequest(BaseModel):
    window_days: int = Field(default=7, ge=1, le=30)
    top_n: int = Field(default=10, ge=1, le=50)


class TopSellerStock(BaseModel):
    product_id: int
    name: str
    revenue: float
    units_sold: int
    stock_qty: int
    low_stock_threshold: int
    stock_status: str


class CheckTopSellersStockResponse(BaseModel):
    window_days: int = 7
    top_sellers: list[TopSellerStock] = Field(default_factory=list)
    out_of_stock_top_sellers: list[TopSellerStock] = Field(default_factory=list)
    low_stock_top_sellers: list[TopSellerStock] = Field(default_factory=list)
    has_stock_issues: bool = False
    potential_revenue_at_risk: float = 0


class InventoryToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_inventory_status(
        self, payload: GetInventoryStatusRequest
    ) -> GetInventoryStatusResponse:
        result = await self._client.invoke("get_inventory_status", payload.model_dump())
        try:
            return GetInventoryStatusResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_inventory_status: {exc}") from exc

    async def predict_stock_out(self, payload: PredictStockOutRequest) -> PredictStockOutResponse:
        result = await self._client.invoke("predict_stock_out", payload.model_dump())
        try:
            return PredictStockOutResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for predict_stock_out: {exc}") from exc

    async def get_low_stock_products(
        self, payload: GetLowStockProductsRequest
    ) -> GetLowStockProductsResponse:
        result = await self._client.invoke("get_low_stock_products", payload.model_dump())
        try:
            return GetLowStockProductsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_low_stock_products: {exc}") from exc

    async def check_top_sellers_stock(
        self, payload: CheckTopSellersStockRequest
    ) -> CheckTopSellersStockResponse:
        result = await self._client.invoke("check_top_sellers_stock", payload.model_dump())
        try:
            return CheckTopSellersStockResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for check_top_sellers_stock: {exc}") from exc

    async def restock_item(self, payload: RestockItemRequest) -> RestockItemResponse:
        """Restock a product (HITL action - calls update_inventory MCP tool)."""
        # Convert to update_inventory payload format
        mcp_payload = {
            "product_id": payload.product_id,
            "quantity_change": payload.quantity,  # Positive for restocking
            "reason": payload.reason or "Restock requested",
        }
        result = await self._client.invoke("update_inventory", mcp_payload)
        try:
            return RestockItemResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for restock_item: {exc}") from exc

    async def update_inventory(self, payload: UpdateInventoryRequest) -> UpdateInventoryResponse:
        """Update product inventory (HITL action)."""
        result = await self._client.invoke("update_inventory", payload.model_dump())
        try:
            return UpdateInventoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for update_inventory: {exc}") from exc
