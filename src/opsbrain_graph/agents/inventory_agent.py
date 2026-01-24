from __future__ import annotations

from typing import Any

from opsbrain_graph.tools import (
    CheckStockRequest,
    PredictStockOutRequest,
    InventoryToolset,
)
from .base_agent import AgentRecommendation, AgentRunContext, AgentTask, BaseAgent


class InventoryAgent(BaseAgent):
    name = "inventory"
    description = "Monitors stock levels and predicts stock-outs."

    async def run(self, task: AgentTask, context: AgentRunContext):
        product_ids = task.parameters.get("product_ids")
        if not product_ids:
            return self.failure("Inventory task requires 'product_ids' parameter.")

        try:
            stock_resp = await self.tools.inventory.check_stock(
                CheckStockRequest(product_ids=product_ids)
            )
        except Exception as exc:
            return self.failure(exc)

        findings = {"stock": [item.model_dump() for item in stock_resp.items]}
        insights = []
        recommendations = []

        for level in stock_resp.items:
            buffer = level.stock_qty - level.low_stock_threshold
            if buffer <= 0:
                insights.append(
                    f"Product {level.product_id} below threshold ({level.stock_qty} <= {level.low_stock_threshold})."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="restock_item",
                        payload={"product_id": level.product_id, "quantity": max(50, -buffer + 10)},
                        reasoning=f"Stock {level.stock_qty} at/below threshold {level.low_stock_threshold}.",
                        requires_approval=True,
                    )
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
