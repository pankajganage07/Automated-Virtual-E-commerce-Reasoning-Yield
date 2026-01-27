from __future__ import annotations

from typing import List, Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


class InventoryStatusPayload(BaseModel):
    product_ids: List[int] = Field(..., min_items=1, max_items=50)


class GetInventoryStatusTool(BaseTool):
    name = "get_inventory_status"

    def request_model(self) -> type[BaseModel]:
        return InventoryStatusPayload

    async def run(self, session, payload: InventoryStatusPayload) -> dict[str, Any]:
        statement = """
            SELECT id, name, stock_qty, low_stock_threshold
            FROM products
            WHERE id = ANY(:ids)
        """
        result = await session.execute(text(statement), {"ids": payload.product_ids})
        rows = [dict(row._mapping) for row in result]
        return {"items": rows}
