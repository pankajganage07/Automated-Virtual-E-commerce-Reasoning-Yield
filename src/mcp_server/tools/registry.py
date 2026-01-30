"""
MCP Tool Registry.

Slimmed architecture:
- Each domain has 2 core tools
- Complex queries route to Data Analyst + HITL
- execute_sql reserved for approved custom queries
"""

from __future__ import annotations

from typing import Dict

from .base import BaseTool
from .sql_tool import ExecuteSQLTool

# Slimmed domain tools (2 per domain)
from .sales_tools import GetSalesSummaryTool, GetTopProductsTool
from .inventory_tool import GetInventoryStatusTool, GetLowStockProductsTool
from .marketing_tool import GetCampaignSpendTool, CalculateROASTool
from .support_tool import GetSupportSentimentTool, GetTicketTrendsTool

from .memory_tool import QueryMemoryTool, SaveMemoryTool, ListIncidentsTool
from .action_tools import (
    UpdateInventoryTool,
    UpdateCampaignStatusTool,
    UpdateCampaignBudgetTool,
    EscalateTicketTool,
    CloseTicketTool,
    PrioritizeTicketTool,
)

TOOL_REGISTRY: Dict[str, BaseTool] = {
    # Data Analyst tool - for approved custom SQL (HITL protected)
    ExecuteSQLTool.name: ExecuteSQLTool(),
    # Query tools - Sales (2 core)
    GetSalesSummaryTool.name: GetSalesSummaryTool(),
    GetTopProductsTool.name: GetTopProductsTool(),
    # Query tools - Inventory (2 core)
    GetInventoryStatusTool.name: GetInventoryStatusTool(),
    GetLowStockProductsTool.name: GetLowStockProductsTool(),
    # Query tools - Marketing (2 core)
    GetCampaignSpendTool.name: GetCampaignSpendTool(),
    CalculateROASTool.name: CalculateROASTool(),
    # Query tools - Support (2 core)
    GetSupportSentimentTool.name: GetSupportSentimentTool(),
    GetTicketTrendsTool.name: GetTicketTrendsTool(),
    # Memory tools
    QueryMemoryTool.name: QueryMemoryTool(),
    SaveMemoryTool.name: SaveMemoryTool(),
    ListIncidentsTool.name: ListIncidentsTool(),
    # Action tools (for HITL execution)
    UpdateInventoryTool.name: UpdateInventoryTool(),
    UpdateCampaignStatusTool.name: UpdateCampaignStatusTool(),
    UpdateCampaignBudgetTool.name: UpdateCampaignBudgetTool(),
    EscalateTicketTool.name: EscalateTicketTool(),
    CloseTicketTool.name: CloseTicketTool(),
    PrioritizeTicketTool.name: PrioritizeTicketTool(),
}
