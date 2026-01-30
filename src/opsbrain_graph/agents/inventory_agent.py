"""
Inventory Agent - Slimmed architecture (2 core capabilities).

Capabilities:
1. check_stock - Check inventory status for products
2. low_stock_scan - Scan for low stock products

Complex queries (stock-out predictions, top-sellers analysis) route to DataAnalystAgent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opsbrain_graph.tools import (
    GetInventoryStatusRequest,
    InventoryToolset,
)
from opsbrain_graph.tools.inventory_tools import (
    GetLowStockProductsRequest,
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


# Query patterns that this agent CANNOT handle (require DataAnalystAgent)
COMPLEX_QUERY_PATTERNS = [
    r"predict.*stock",
    r"stock.*out.*predict",
    r"when.*run\s*out",
    r"forecast.*inventory",
    r"top.*(seller|product).*stock",
    r"best.*(seller|product).*(out|low)",
    r"compare.*inventory.*period",
    r"historical.*stock",
    r"trend.*inventory",
    r"stock.*trend",
    r"regional.*inventory",
    r"warehouse.*comparison",
    r"turnover.*rate",
    r"inventory.*velocity",
    r"days.*of.*supply",
]


class InventoryAgent(BaseAgent):
    """
    Inventory Agent with 2 core capabilities.

    Complex queries trigger cannot_handle for routing to DataAnalystAgent.
    """

    name = "inventory"
    description = "Monitors stock levels and identifies low-stock items."

    metadata = AgentMetadata(
        name="inventory",
        display_name="INVENTORY",
        description="Monitors current stock levels and identifies low-stock products. For complex analytics (predictions, top-seller analysis), use data analyst.",
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
        ],
        keywords=[
            "stock",
            "inventory",
            "out of stock",
            "restock",
            "low stock",
            "quantity",
        ],
        priority_boost=["out of stock", "urgent restock", "stockout"],
    )

    def _is_complex_query(self, query: str) -> bool:
        """Check if query requires complex analysis."""
        query_lower = query.lower()
        for pattern in COMPLEX_QUERY_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _cannot_handle(self, query: str) -> AgentResult:
        """Return cannot_handle status for supervisor to route to analyst."""
        return AgentResult(
            status="cannot_handle",
            findings={
                "query": query,
                "reason": "This query requires complex inventory analysis (prediction, trending, cross-analysis) that needs custom SQL.",
                "suggested_agent": "data_analyst",
            },
            insights=[
                "This inventory query requires advanced analytics beyond my core capabilities.",
                "Routing to Data Analyst for custom SQL generation with HITL approval.",
            ],
            recommendations=[],
        )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        query = params.get("query", "")
        mode = params.get("mode", "check_stock")

        # Check for complex queries first
        if self._is_complex_query(query):
            logger.info("inventory agent: complex query detected, returning cannot_handle")
            return self._cannot_handle(query)

        if mode == "low_stock_scan":
            return await self._run_low_stock_scan(params)
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
