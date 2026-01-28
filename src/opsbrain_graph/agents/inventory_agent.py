from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.tools import (
    GetInventoryStatusRequest,
    PredictStockOutRequest,
    InventoryToolset,
)
from opsbrain_graph.tools.inventory_tools import (
    GetLowStockProductsRequest,
    CheckTopSellersStockRequest,
)
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentRunContext,
    AgentTask,
    BaseAgent,
    AgentResult,
)

logger = logging.getLogger("agent.inventory")


class InventoryAgent(BaseAgent):
    name = "inventory"
    description = "Monitors stock levels and predicts stock-outs."

    metadata = AgentMetadata(
        name="inventory",
        display_name="INVENTORY",
        description="Monitors stock levels, identifies low-stock items, predicts stock-outs, and checks if top sellers have inventory issues.",
        capabilities=[
            AgentCapability(
                name="check_stock",
                description="Check current stock levels for specific products",
                parameters={
                    "product_ids": "List of product IDs to check (required)",
                },
                example_queries=[
                    "What's the stock level for product 123?",
                    "Check inventory for our top sellers",
                ],
            ),
            AgentCapability(
                name="low_stock_alert",
                description="Identify products below their low_stock_threshold",
                parameters={
                    "product_ids": "List of product IDs to check (optional - if not provided, checks all)",
                },
                example_queries=[
                    "Which products need restocking?",
                    "Show me low stock items",
                    "What's about to run out?",
                ],
            ),
            AgentCapability(
                name="low_stock_scan",
                description="Scan ALL products for low stock issues without needing product IDs",
                parameters={
                    "include_out_of_stock": "Include completely out-of-stock items (default: true)",
                    "limit": "Max products to return (default: 20)",
                },
                example_queries=[
                    "Which products are close to stock-out?",
                    "Are we running low on any products?",
                    "Show all low inventory items",
                ],
            ),
            AgentCapability(
                name="top_sellers_stock",
                description="Check if top-selling products have stock issues",
                parameters={
                    "window_days": "Period to check top sellers (default: 7)",
                    "top_n": "Number of top sellers to check (default: 10)",
                },
                example_queries=[
                    "Were any top-selling products out of stock yesterday?",
                    "Do our best sellers have enough inventory?",
                    "Check stock levels for top performers",
                ],
            ),
        ],
        keywords=[
            "stock",
            "inventory",
            "out of stock",
            "restock",
            "supply",
            "warehouse",
            "quantity",
            "low stock",
            "stockout",
        ],
        priority_boost=["out of stock", "urgent restock", "stockout"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "check_stock")

        if mode == "low_stock_scan":
            return await self._run_low_stock_scan(params)
        elif mode == "top_sellers_stock":
            return await self._run_top_sellers_stock(params)
        else:
            return await self._run_check_stock(params)

    async def _run_check_stock(self, params: dict[str, Any]) -> AgentResult:
        """Check stock for specific products."""
        product_ids = params.get("product_ids")
        if not product_ids:
            # Fall back to low_stock_scan if no product_ids provided
            return await self._run_low_stock_scan(params)

        try:
            stock_resp = await self.tools.inventory.get_inventory_status(
                GetInventoryStatusRequest(product_ids=product_ids)
            )
        except Exception as exc:
            logger.exception("inventory agent (check_stock) failed: %s", exc)
            return self.failure(exc)

        findings = {"stock": [item.model_dump() for item in stock_resp.items]}
        insights = []
        recommendations = []

        for item in stock_resp.items:
            buffer = item.stock_qty - item.low_stock_threshold
            if buffer <= 0:
                insights.append(
                    f"⚠️ Product {item.name} (ID: {item.id}) below threshold ({item.stock_qty} <= {item.low_stock_threshold})."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="restock_item",
                        payload={"product_id": item.id, "quantity": max(50, -buffer + 10)},
                        reasoning=f"Stock {item.stock_qty} at/below threshold {item.low_stock_threshold}.",
                        requires_approval=True,
                    )
                )
            else:
                insights.append(
                    f"✅ Product {item.name} (ID: {item.id}): {item.stock_qty} in stock (buffer: {buffer})"
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_low_stock_scan(self, params: dict[str, Any]) -> AgentResult:
        """Scan all products for low stock issues."""
        include_out_of_stock = params.get("include_out_of_stock", True)
        limit = params.get("limit", 20)

        try:
            resp = await self.tools.inventory.get_low_stock_products(
                GetLowStockProductsRequest(
                    include_out_of_stock=include_out_of_stock,
                    limit=limit,
                )
            )
        except Exception as exc:
            logger.exception("inventory agent (low_stock_scan) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "low_stock_products": [p.model_dump() for p in resp.low_stock_products],
            "total_count": resp.total_count,
            "out_of_stock_count": resp.out_of_stock_count,
            "critical_count": resp.critical_count,
            "has_critical": resp.has_critical,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        if resp.total_count == 0:
            insights.append("✅ All products are well-stocked. No low inventory issues detected.")
        else:
            insights.append(f"Found {resp.total_count} products with low stock:")
            if resp.out_of_stock_count > 0:
                insights.append(f"  🔴 {resp.out_of_stock_count} products completely OUT OF STOCK")
            if resp.critical_count > 0:
                insights.append(f"  🟠 {resp.critical_count} products in CRITICAL stock level")

            for product in resp.low_stock_products[:10]:
                status_icon = (
                    "🔴"
                    if product.status == "out_of_stock"
                    else "🟠" if product.status == "critical" else "🟡"
                )
                insights.append(
                    f"  {status_icon} {product.name}: {product.stock_qty} units (threshold: {product.low_stock_threshold})"
                )

                if product.needs_restock:
                    restock_qty = max(50, product.low_stock_threshold - product.stock_qty + 20)
                    recommendations.append(
                        AgentRecommendation(
                            action_type="restock_item",
                            payload={"product_id": product.product_id, "quantity": restock_qty},
                            reasoning=f"{product.name} has {product.stock_qty} units, below threshold of {product.low_stock_threshold}",
                            requires_approval=True,
                        )
                    )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_top_sellers_stock(self, params: dict[str, Any]) -> AgentResult:
        """Check if top-selling products have stock issues."""
        window_days = params.get("window_days", 7)
        top_n = params.get("top_n", 10)

        try:
            resp = await self.tools.inventory.check_top_sellers_stock(
                CheckTopSellersStockRequest(window_days=window_days, top_n=top_n)
            )
        except Exception as exc:
            logger.exception("inventory agent (top_sellers_stock) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "window_days": resp.window_days,
            "top_sellers": [s.model_dump() for s in resp.top_sellers],
            "out_of_stock_top_sellers": [s.model_dump() for s in resp.out_of_stock_top_sellers],
            "low_stock_top_sellers": [s.model_dump() for s in resp.low_stock_top_sellers],
            "has_stock_issues": resp.has_stock_issues,
            "potential_revenue_at_risk": resp.potential_revenue_at_risk,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"Top {top_n} sellers stock check (last {window_days} days):")

        if not resp.has_stock_issues:
            insights.append("✅ All top-selling products are well-stocked!")
        else:
            if resp.out_of_stock_top_sellers:
                insights.append(
                    f"🔴 {len(resp.out_of_stock_top_sellers)} top sellers are OUT OF STOCK:"
                )
                for s in resp.out_of_stock_top_sellers:
                    insights.append(
                        f"    - {s.name} (${s.revenue:,.2f} revenue, {s.units_sold} sold)"
                    )
                    recommendations.append(
                        AgentRecommendation(
                            action_type="urgent_restock",
                            payload={
                                "product_id": s.product_id,
                                "quantity": max(100, s.units_sold),
                            },
                            reasoning=f"Top seller {s.name} is out of stock - generated ${s.revenue:,.2f} in revenue",
                            requires_approval=True,
                        )
                    )

            if resp.low_stock_top_sellers:
                insights.append(f"🟠 {len(resp.low_stock_top_sellers)} top sellers have LOW stock:")
                for s in resp.low_stock_top_sellers:
                    insights.append(f"    - {s.name}: {s.stock_qty} units left")

            if resp.potential_revenue_at_risk > 0:
                insights.append(
                    f"💰 Potential revenue at risk: ${resp.potential_revenue_at_risk:,.2f}"
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
