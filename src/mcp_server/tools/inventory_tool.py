from __future__ import annotations

from typing import List, Any
from datetime import datetime, timedelta

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


# =============================================================================
# PREDICT STOCK OUT TOOL
# =============================================================================


class PredictStockOutPayload(BaseModel):
    product_id: int = Field(..., description="Product ID to analyze")
    lookahead_days: int = Field(default=7, ge=1, le=30, description="Days to look ahead")


class PredictStockOutTool(BaseTool):
    """
    Predicts when a product will run out of stock based on recent sales velocity.
    """

    name = "predict_stock_out"

    def request_model(self) -> type[BaseModel]:
        return PredictStockOutPayload

    async def run(self, session, payload: PredictStockOutPayload) -> dict[str, Any]:
        # Get current stock and product info
        product_stmt = text(
            """
            SELECT id, name, stock_qty, low_stock_threshold
            FROM products
            WHERE id = :product_id
        """
        )
        product_result = await session.execute(product_stmt, {"product_id": payload.product_id})
        product = product_result.one_or_none()

        if product is None:
            return {
                "product_id": payload.product_id,
                "error": f"Product {payload.product_id} not found",
                "projected_stockout_date": None,
                "confidence": 0.0,
            }

        current_stock = product.stock_qty

        # Calculate average daily sales over the last 14 days
        velocity_stmt = text(
            """
            SELECT 
                COALESCE(SUM(qty), 0) AS total_sold,
                COUNT(DISTINCT DATE(timestamp)) AS days_with_sales
            FROM orders
            WHERE product_id = :product_id
              AND timestamp >= NOW() - INTERVAL '14 days'
        """
        )
        velocity_result = await session.execute(velocity_stmt, {"product_id": payload.product_id})
        velocity_row = velocity_result.one()

        total_sold = velocity_row.total_sold or 0
        days_with_sales = velocity_row.days_with_sales or 1

        # Calculate daily velocity
        avg_daily_sales = total_sold / 14.0  # Average over full 14-day window

        if avg_daily_sales <= 0:
            return {
                "product_id": payload.product_id,
                "product_name": product.name,
                "current_stock": current_stock,
                "avg_daily_sales": 0.0,
                "projected_stockout_date": None,
                "days_until_stockout": None,
                "confidence": 0.3,  # Low confidence due to no sales data
                "risk_level": "unknown",
                "message": "No recent sales data to predict stock-out",
            }

        # Calculate days until stock-out
        days_until_stockout = current_stock / avg_daily_sales
        projected_date = datetime.utcnow() + timedelta(days=days_until_stockout)

        # Calculate confidence based on data quality
        confidence = min(0.95, 0.5 + (days_with_sales / 14.0) * 0.45)

        # Determine risk level
        if days_until_stockout <= 3:
            risk_level = "critical"
        elif days_until_stockout <= 7:
            risk_level = "high"
        elif days_until_stockout <= 14:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Check if within lookahead window
        within_window = days_until_stockout <= payload.lookahead_days

        return {
            "product_id": payload.product_id,
            "product_name": product.name,
            "current_stock": current_stock,
            "low_stock_threshold": product.low_stock_threshold,
            "avg_daily_sales": round(avg_daily_sales, 2),
            "days_until_stockout": round(days_until_stockout, 1),
            "projected_stockout_date": projected_date.isoformat() if within_window else None,
            "within_lookahead_window": within_window,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
        }
