from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .exceptions import MCPError
from .mcp_client import MCPClient


class CheckStockRequest(BaseModel):
    product_ids: list[int] = Field(..., min_length=1)


class StockLevel(BaseModel):
    product_id: int
    stock_qty: int
    low_stock_threshold: int
    incoming_qty: int | None = None
    eta_days: int | None = None


class CheckStockResponse(BaseModel):
    items: list[StockLevel]


class PredictStockOutRequest(BaseModel):
    product_id: int
    lookahead_days: int = 7


class PredictStockOutResponse(BaseModel):
    product_id: int
    projected_stockout_date: str | None = None
    confidence: float | None = None


class RestockItemRequest(BaseModel):
    product_id: int
    quantity: int
    priority: str = "normal"


class RestockItemResponse(BaseModel):
    pending_action_id: str
    message: str


class InventoryToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def check_stock(self, payload: CheckStockRequest) -> CheckStockResponse:
        result = await self._client.invoke("check_stock", payload.model_dump())
        try:
            return CheckStockResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for check_stock: {exc}") from exc

    async def predict_stock_out(self, payload: PredictStockOutRequest) -> PredictStockOutResponse:
        result = await self._client.invoke("predict_stock_out", payload.model_dump())
        try:
            return PredictStockOutResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for predict_stock_out: {exc}") from exc

    async def restock_item(self, payload: RestockItemRequest) -> RestockItemResponse:
        result = await self._client.invoke("restock_item", payload.model_dump())
        try:
            return RestockItemResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for restock_item: {exc}") from exc
