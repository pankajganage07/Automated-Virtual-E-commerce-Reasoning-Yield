"""
Inventory tools for LangGraph agents.

Slimmed toolset (2 core tools):
1. get_inventory_status - Get stock levels for products
2. get_low_stock_products - Find products below threshold

Complex queries should route to DataAnalystAgent with HITL.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


# =============================================================================
# GET INVENTORY STATUS (Core Tool #1)
# =============================================================================


class GetInventoryStatusRequest(BaseModel):
    product_ids: list[int] | None = Field(
        default=None, description="Product IDs to check (optional)"
    )
    limit: int = Field(default=50, ge=1, le=200)


class InventoryItem(BaseModel):
    product_id: int
    name: str
    category: str | None = None
    stock_qty: int
    low_stock_threshold: int
    status: str  # out_of_stock, low_stock, in_stock


class GetInventoryStatusResponse(BaseModel):
    items: list[InventoryItem] = Field(default_factory=list)
    total_count: int = 0
    out_of_stock_count: int = 0
    low_stock_count: int = 0


# =============================================================================
# GET LOW STOCK PRODUCTS (Core Tool #2)
# =============================================================================


class GetLowStockProductsRequest(BaseModel):
    include_out_of_stock: bool = Field(default=True)
    category: str | None = Field(default=None, description="Optional category filter")
    limit: int = Field(default=20, ge=1, le=100)


class LowStockProduct(BaseModel):
    product_id: int
    name: str
    category: str | None = None
    stock_qty: int
    low_stock_threshold: int
    buffer: int
    status: str  # out_of_stock, critical, warning
    needs_restock: bool = True


class GetLowStockProductsResponse(BaseModel):
    low_stock_products: list[LowStockProduct] = Field(default_factory=list)
    total_count: int = 0
    out_of_stock_count: int = 0
    critical_count: int = 0
    has_critical: bool = False


# =============================================================================
# ACTION TOOLS (for HITL execution)
# =============================================================================


class UpdateInventoryRequest(BaseModel):
    """Request to update inventory (HITL action)."""

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


# =============================================================================
# INVENTORY TOOLSET
# =============================================================================


class InventoryToolset:
    """
    Inventory tools (2 core tools).

    For complex queries (stock predictions, top seller stock checks),
    the agent should indicate it cannot handle the query, and the supervisor
    will route to the DataAnalystAgent.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def get_inventory_status(
        self, payload: GetInventoryStatusRequest
    ) -> GetInventoryStatusResponse:
        """Get stock levels for products."""
        result = await self._client.invoke("get_inventory_status", payload.model_dump())
        try:
            return GetInventoryStatusResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_inventory_status: {exc}") from exc

    async def get_low_stock_products(
        self, payload: GetLowStockProductsRequest
    ) -> GetLowStockProductsResponse:
        """Find products below their stock threshold."""
        result = await self._client.invoke("get_low_stock_products", payload.model_dump())
        try:
            return GetLowStockProductsResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for get_low_stock_products: {exc}") from exc

    async def update_inventory(self, payload: UpdateInventoryRequest) -> UpdateInventoryResponse:
        """Update product inventory (HITL action)."""
        result = await self._client.invoke("update_inventory", payload.model_dump())
        try:
            return UpdateInventoryResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for update_inventory: {exc}") from exc
