"""
Inventory Agent - Monitors stock levels.

Capabilities:
1. check_stock - Check inventory status for products
2. low_stock_scan - Scan for low stock products
"""

from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.tools import (
    GetInventoryStatusRequest,
)
from opsbrain_graph.tools.inventory_tools import (
    GetLowStockProductsRequest,
    SearchProductsRequest,
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
    """Inventory Agent with 3 core capabilities."""

    name = "inventory"
    description = "Monitors stock levels, identifies low-stock items, and handles restock requests."

    metadata = AgentMetadata(
        name="inventory",
        display_name="INVENTORY",
        description="Monitors current stock levels, identifies low-stock products, and handles restock requests.",
        capabilities=[
            AgentCapability(
                name="check_stock",
                description="Check current stock levels for specific products",
                parameters={
                    "product_ids": "List of product IDs to check (optional)",
                },
                example_queries=[
                    "What's the stock level for product 123?",
                    "Check inventory for these products",
                ],
            ),
            AgentCapability(
                name="low_stock_scan",
                description="Scan ALL products for low stock issues",
                parameters={
                    "include_out_of_stock": "Include out-of-stock items (default: true)",
                    "limit": "Max products to return (default: 20)",
                },
                example_queries=[
                    "Which products need restocking?",
                    "Show me low stock items",
                    "What's about to run out?",
                ],
            ),
            AgentCapability(
                name="restock",
                description="Request a restock for a specific product by name or ID",
                parameters={
                    "product_name": "Name of the product to restock",
                    "product_id": "ID of the product to restock (optional if name provided)",
                    "quantity": "Quantity to restock (default: 50)",
                },
                example_queries=[
                    "Restock the product EcoWater Bottle",
                    "Can you restock product 10?",
                    "Please add 100 units of Widget Pro to inventory",
                    "Restock EcoWater Bottle in inventory",
                ],
            ),
        ],
        keywords=[
            "stock",
            "inventory",
            "out of stock",
            "restock",
            "low stock",
            "quantity",
            "add units",
            "replenish",
        ],
        priority_boost=["out of stock", "urgent restock", "stockout", "restock"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        """Execute the inventory agent task based on mode."""
        params = task.parameters
        mode = params.get("mode", "check_stock")

        if mode == "low_stock_scan":
            return await self._run_low_stock_scan(params)
        elif mode == "restock":
            return await self._run_restock(params)
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
                    f"⚠️ Product {item.name} (ID: {item.product_id}) below threshold ({item.stock_qty} <= {item.low_stock_threshold})."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="restock_item",
                        payload={"product_id": item.product_id, "quantity": max(50, -buffer + 10)},
                        reasoning=f"Stock {item.stock_qty} at/below threshold {item.low_stock_threshold}.",
                        requires_approval=True,
                    )
                )
            else:
                insights.append(
                    f"✅ Product {item.name} (ID: {item.product_id}): {item.stock_qty} in stock (buffer: {buffer})"
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

    async def _run_restock(self, params: dict[str, Any]) -> AgentResult:
        """Handle direct restock request for a product by name or ID."""
        product_name = params.get("product_name")
        product_id = params.get("product_id")
        quantity = params.get("quantity", 50)

        product = None

        # Use the search_products tool which directly queries the products table
        try:
            if product_id:
                # Search by product ID
                search_resp = await self.tools.inventory.search_products(
                    SearchProductsRequest(product_id=product_id)
                )
                if search_resp.products:
                    product = search_resp.products[0]
                else:
                    return self.success(
                        findings={"error": f"Product ID {product_id} not found"},
                        insights=[f"❌ Could not find product with ID {product_id}"],
                        recommendations=[],
                    )
            elif product_name:
                # Search by product name (case-insensitive partial match)
                search_resp = await self.tools.inventory.search_products(
                    SearchProductsRequest(product_name=product_name)
                )
                if search_resp.products:
                    # Use the first match
                    product = search_resp.products[0]
                    product_id = product.product_id

                    # If multiple matches, show them in insights
                    if len(search_resp.products) > 1:
                        logger.info(
                            "Found %d products matching '%s', using first: %s",
                            len(search_resp.products),
                            product_name,
                            product.name,
                        )
                else:
                    return self.success(
                        findings={"error": f"Product '{product_name}' not found"},
                        insights=[f"❌ Could not find product matching '{product_name}'"],
                        recommendations=[],
                    )
            else:
                return self.success(
                    findings={"error": "No product_name or product_id provided"},
                    insights=["❌ Please specify a product name or ID to restock"],
                    recommendations=[],
                )
        except Exception as exc:
            logger.exception("Failed to search for product: %s", exc)
            return self.failure(exc)

        # Create restock recommendation
        findings = {
            "product_id": product.product_id,
            "product_name": product.name,
            "current_stock": product.stock_qty,
            "restock_quantity": quantity,
        }
        insights = [
            f"📦 Restock request for **{product.name}** (ID: {product.product_id})",
            f"   Current stock: {product.stock_qty} units",
            f"   Requested restock quantity: {quantity} units",
        ]
        recommendations = [
            AgentRecommendation(
                action_type="restock_item",
                payload={"product_id": product.product_id, "quantity": quantity},
                reasoning=f"User requested restock of {quantity} units for {product.name}",
                requires_approval=True,
            )
        ]

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
