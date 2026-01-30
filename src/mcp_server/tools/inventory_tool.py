"""
MCP Tools for inventory domain.

Core tools (2 main):
1. get_inventory_status - Get stock levels for specific products
2. get_low_stock_products - Find products below threshold

Complex queries (stock predictions, top seller stock checks)
should be routed to the Data Analyst agent with HITL approval.
"""

from __future__ import annotations

from typing import List, Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


# =============================================================================
# GET INVENTORY STATUS TOOL
# =============================================================================


class InventoryStatusPayload(BaseModel):
    product_ids: List[int] | None = Field(
        default=None, description="Product IDs to check (optional, returns all if not specified)"
    )
    limit: int = Field(default=50, ge=1, le=200)


class GetInventoryStatusTool(BaseTool):
    """Return current stock levels for products."""

    name = "get_inventory_status"

    def request_model(self) -> type[BaseModel]:
        return InventoryStatusPayload

    async def run(self, session, payload: InventoryStatusPayload) -> dict[str, Any]:
        # Build query dynamically to avoid NULL parameter type issues with asyncpg
        params = {"limit": payload.limit}
        where_clause = ""

        if payload.product_ids is not None:
            where_clause = "WHERE id = ANY(:ids)"
            params["ids"] = payload.product_ids

        statement = f"""
            SELECT 
                id, 
                name, 
                category,
                stock_qty, 
                low_stock_threshold,
                CASE 
                    WHEN stock_qty = 0 THEN 'out_of_stock'
                    WHEN stock_qty <= low_stock_threshold THEN 'low_stock'
                    ELSE 'in_stock'
                END AS status
            FROM products
            {where_clause}
            ORDER BY stock_qty ASC
            LIMIT :limit
        """
        result = await session.execute(text(statement), params)

        items = []
        out_of_stock_count = 0
        low_stock_count = 0

        for row in result:
            item = {
                "product_id": row.id,
                "name": row.name,
                "category": row.category,
                "stock_qty": row.stock_qty,
                "low_stock_threshold": row.low_stock_threshold,
                "status": row.status,
            }
            items.append(item)

            if row.status == "out_of_stock":
                out_of_stock_count += 1
            elif row.status == "low_stock":
                low_stock_count += 1

        return {
            "items": items,
            "total_count": len(items),
            "out_of_stock_count": out_of_stock_count,
            "low_stock_count": low_stock_count,
        }


# =============================================================================
# GET LOW STOCK PRODUCTS TOOL
# =============================================================================


class LowStockProductsPayload(BaseModel):
    """Find all products below their stock threshold."""

    include_out_of_stock: bool = Field(default=True, description="Include products with 0 stock")
    category: str | None = Field(default=None, description="Optional category filter")
    limit: int = Field(default=20, ge=1, le=100)


class GetLowStockProductsTool(BaseTool):
    """
    Find all products that are at or below their low_stock_threshold.
    Does NOT require product_ids - scans entire inventory.
    """

    name = "get_low_stock_products"

    def request_model(self) -> type[BaseModel]:
        return LowStockProductsPayload

    async def run(self, session, payload: LowStockProductsPayload) -> dict[str, Any]:
        # Build query dynamically to avoid asyncpg NULL parameter type issues
        params = {"limit": payload.limit}

        # Build WHERE conditions
        where_conditions = []

        if payload.include_out_of_stock:
            where_conditions.append("(p.stock_qty <= p.low_stock_threshold OR p.stock_qty = 0)")
        else:
            where_conditions.append("(p.stock_qty <= p.low_stock_threshold AND p.stock_qty > 0)")

        if payload.category is not None:
            where_conditions.append("p.category = :category")
            params["category"] = payload.category

        where_clause = " AND ".join(where_conditions)

        statement = f"""
            SELECT 
                p.id,
                p.name,
                p.category,
                p.stock_qty,
                p.low_stock_threshold,
                p.stock_qty - p.low_stock_threshold AS buffer,
                CASE 
                    WHEN p.stock_qty = 0 THEN 'out_of_stock'
                    WHEN p.stock_qty <= p.low_stock_threshold THEN 'critical'
                    WHEN p.stock_qty <= p.low_stock_threshold * 1.5 THEN 'warning'
                    ELSE 'ok'
                END AS status
            FROM products p
            WHERE {where_clause}
            ORDER BY buffer ASC, p.stock_qty ASC
            LIMIT :limit
        """
        result = await session.execute(text(statement), params)

        products = []
        out_of_stock_count = 0
        critical_count = 0

        for row in result:
            product = {
                "product_id": row.id,
                "name": row.name,
                "category": row.category,
                "stock_qty": row.stock_qty,
                "low_stock_threshold": row.low_stock_threshold,
                "buffer": row.buffer,
                "status": row.status,
                "needs_restock": True,
            }
            products.append(product)

            if row.stock_qty == 0:
                out_of_stock_count += 1
            if row.status == "critical":
                critical_count += 1

        return {
            "low_stock_products": products,
            "total_count": len(products),
            "out_of_stock_count": out_of_stock_count,
            "critical_count": critical_count,
            "has_critical": critical_count > 0 or out_of_stock_count > 0,
        }
