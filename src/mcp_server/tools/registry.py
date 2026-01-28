from __future__ import annotations

from typing import Dict

from .base import BaseTool
from .sql_tool import ExecuteSQLTool
from .sales_tools import (
    GetSalesSummaryTool,
    GetTopProductsTool,
    CompareSalesPeriodsTool,
    GetRegionalSalesTool,
    GetChannelPerformanceTool,
    GetProductContributionTool,
)
from .inventory_tool import (
    GetInventoryStatusTool,
    PredictStockOutTool,
    GetLowStockProductsTool,
    CheckTopSellersStockTool,
)
from .marketing_tool import (
    GetCampaignSpendTool,
    CalculateROASTool,
    GetUnderperformingCampaignsTool,
    CompareCampaignPerformanceTool,
)
from .support_tool import (
    GetSupportSentimentTool,
    GetTicketTrendsTool,
    GetCommonIssuesTool,
    GetComplaintTrendsTool,
)
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
    # Query tools - Sales
    ExecuteSQLTool.name: ExecuteSQLTool(),
    GetSalesSummaryTool.name: GetSalesSummaryTool(),
    GetTopProductsTool.name: GetTopProductsTool(),
    CompareSalesPeriodsTool.name: CompareSalesPeriodsTool(),
    GetRegionalSalesTool.name: GetRegionalSalesTool(),
    GetChannelPerformanceTool.name: GetChannelPerformanceTool(),
    GetProductContributionTool.name: GetProductContributionTool(),
    # Query tools - Inventory
    GetInventoryStatusTool.name: GetInventoryStatusTool(),
    PredictStockOutTool.name: PredictStockOutTool(),
    GetLowStockProductsTool.name: GetLowStockProductsTool(),
    CheckTopSellersStockTool.name: CheckTopSellersStockTool(),
    # Query tools - Marketing
    GetCampaignSpendTool.name: GetCampaignSpendTool(),
    CalculateROASTool.name: CalculateROASTool(),
    GetUnderperformingCampaignsTool.name: GetUnderperformingCampaignsTool(),
    CompareCampaignPerformanceTool.name: CompareCampaignPerformanceTool(),
    # Query tools - Support
    GetSupportSentimentTool.name: GetSupportSentimentTool(),
    GetTicketTrendsTool.name: GetTicketTrendsTool(),
    GetCommonIssuesTool.name: GetCommonIssuesTool(),
    GetComplaintTrendsTool.name: GetComplaintTrendsTool(),
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
