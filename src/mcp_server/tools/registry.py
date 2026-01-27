from __future__ import annotations

from typing import Dict

from .base import BaseTool
from .sql_tool import ExecuteSQLTool
from .sales_tools import GetSalesSummaryTool, GetTopProductsTool
from .inventory_tool import GetInventoryStatusTool, PredictStockOutTool
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
    # Query tools
    ExecuteSQLTool.name: ExecuteSQLTool(),
    GetSalesSummaryTool.name: GetSalesSummaryTool(),
    GetTopProductsTool.name: GetTopProductsTool(),
    GetInventoryStatusTool.name: GetInventoryStatusTool(),
    GetCampaignSpendTool.name: GetCampaignSpendTool(),
    GetSupportSentimentTool.name: GetSupportSentimentTool(),
    # Memory tools
    QueryMemoryTool.name: QueryMemoryTool(),
    SaveMemoryTool.name: SaveMemoryTool(),
    ListIncidentsTool.name: ListIncidentsTool(),
    # Analysis tools
    PredictStockOutTool.name: PredictStockOutTool(),
    CalculateROASTool.name: CalculateROASTool(),
    GetTicketTrendsTool.name: GetTicketTrendsTool(),
    # Action tools (for HITL execution)
    UpdateInventoryTool.name: UpdateInventoryTool(),
    UpdateCampaignStatusTool.name: UpdateCampaignStatusTool(),
    UpdateCampaignBudgetTool.name: UpdateCampaignBudgetTool(),
    EscalateTicketTool.name: EscalateTicketTool(),
    CloseTicketTool.name: CloseTicketTool(),
    PrioritizeTicketTool.name: PrioritizeTicketTool(),
}
