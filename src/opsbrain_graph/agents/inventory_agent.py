from __future__ import annotations

from typing import Any

from opsbrain_graph.tools import (
    GetInventoryStatusRequest,
    PredictStockOutRequest,
    InventoryToolset,
)
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)


class InventoryAgent(BaseAgent):
    name = "inventory"
    description = "Monitors stock levels and predicts stock-outs."

    metadata = AgentMetadata(
        name="inventory",
        display_name="INVENTORY",
        description="Monitors stock levels, identifies low-stock items, and predicts potential stock-outs. Can recommend restocking actions.",
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
                    "Are we running low on any products?",
                ],
            ),
            AgentCapability(
                name="low_stock_alert",
                description="Identify products below their low_stock_threshold",
                parameters={
                    "product_ids": "List of product IDs to check",
                },
                example_queries=[
                    "Which products need restocking?",
                    "Show me low stock items",
                    "What's about to run out?",
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
        ],
        priority_boost=["out of stock", "urgent restock", "stockout"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext):
        product_ids = task.parameters.get("product_ids")
        if not product_ids:
            return self.failure("Inventory task requires 'product_ids' parameter.")

        try:
            stock_resp = await self.tools.inventory.get_inventory_status(
                GetInventoryStatusRequest(product_ids=product_ids)
            )
        except Exception as exc:
            return self.failure(exc)

        findings = {"stock": [item.model_dump() for item in stock_resp.items]}
        insights = []
        recommendations = []

        for item in stock_resp.items:
            buffer = item.stock_qty - item.low_stock_threshold
            if buffer <= 0:
                insights.append(
                    f"Product {item.name} (ID: {item.id}) below threshold ({item.stock_qty} <= {item.low_stock_threshold})."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="restock_item",
                        payload={"product_id": item.id, "quantity": max(50, -buffer + 10)},
                        reasoning=f"Stock {item.stock_qty} at/below threshold {item.low_stock_threshold}.",
                        requires_approval=True,
                    )
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
