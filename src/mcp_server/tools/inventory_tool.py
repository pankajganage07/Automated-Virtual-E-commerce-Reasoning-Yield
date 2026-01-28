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


# =============================================================================
# GET LOW STOCK PRODUCTS TOOL
# =============================================================================


class LowStockProductsPayload(BaseModel):
    """Find all products below their stock threshold."""

    include_out_of_stock: bool = Field(default=True, description="Include products with 0 stock")
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
        # Get all products below threshold
        stmt = text(
            """
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
            WHERE p.stock_qty <= p.low_stock_threshold
               OR (:include_out_of_stock AND p.stock_qty = 0)
            ORDER BY buffer ASC, p.stock_qty ASC
            LIMIT :limit
        """
        )
        result = await session.execute(
            stmt, {"include_out_of_stock": payload.include_out_of_stock, "limit": payload.limit}
        )

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


# =============================================================================
# CHECK TOP SELLERS STOCK TOOL
# =============================================================================


class TopSellersStockPayload(BaseModel):
    """Check if top selling products have stock issues."""

    window_days: int = Field(default=7, ge=1, le=30)
    top_n: int = Field(default=10, ge=1, le=50)


class CheckTopSellersStockTool(BaseTool):
    """
    Check if any top-selling products are out of stock or low on stock.
    Answers: "Were any top-selling products out of stock?"
    """

    name = "check_top_sellers_stock"

    def request_model(self) -> type[BaseModel]:
        return TopSellersStockPayload

    async def run(self, session, payload: TopSellersStockPayload) -> dict[str, Any]:
        # Get top sellers with their current stock
        stmt = text(
            """
            SELECT
                p.id,
                p.name,
                p.stock_qty,
                p.low_stock_threshold,
                COALESCE(SUM(o.revenue), 0) AS total_revenue,
                COALESCE(SUM(o.qty), 0) AS units_sold,
                CASE 
                    WHEN p.stock_qty = 0 THEN 'out_of_stock'
                    WHEN p.stock_qty <= p.low_stock_threshold THEN 'low_stock'
                    ELSE 'in_stock'
                END AS stock_status
            FROM products p
            LEFT JOIN orders o ON o.product_id = p.id
                AND o.timestamp >= NOW() - INTERVAL :window_days || ' days'
            GROUP BY p.id, p.name, p.stock_qty, p.low_stock_threshold
            HAVING COALESCE(SUM(o.revenue), 0) > 0
            ORDER BY total_revenue DESC
            LIMIT :top_n
        """
        )
        result = await session.execute(
            stmt, {"window_days": payload.window_days, "top_n": payload.top_n}
        )

        products = []
        out_of_stock = []
        low_stock = []

        for row in result:
            product = {
                "product_id": row.id,
                "name": row.name,
                "revenue": round(float(row.total_revenue), 2),
                "units_sold": row.units_sold,
                "stock_qty": row.stock_qty,
                "low_stock_threshold": row.low_stock_threshold,
                "stock_status": row.stock_status,
            }
            products.append(product)

            if row.stock_status == "out_of_stock":
                out_of_stock.append(product)
            elif row.stock_status == "low_stock":
                low_stock.append(product)

        # Calculate potential lost revenue
        potential_lost_revenue = sum(p["revenue"] for p in out_of_stock)

        return {
            "window_days": payload.window_days,
            "top_sellers": products,
            "out_of_stock_top_sellers": out_of_stock,
            "low_stock_top_sellers": low_stock,
            "has_stock_issues": len(out_of_stock) > 0 or len(low_stock) > 0,
            "potential_revenue_at_risk": round(potential_lost_revenue, 2),
        }
