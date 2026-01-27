from __future__ import annotations

from typing import Dict

from .base import BaseTool
from .sql_tool import ExecuteSQLTool
from .sales_tools import GetSalesSummaryTool, GetTopProductsTool
from .inventory_tool import GetInventoryStatusTool
from .marketing_tool import GetCampaignSpendTool
from .support_tool import GetSupportSentimentTool
from .memory_tool import QueryMemoryTool, SaveMemoryTool

TOOL_REGISTRY: Dict[str, BaseTool] = {
    ExecuteSQLTool.name: ExecuteSQLTool(),
    GetSalesSummaryTool.name: GetSalesSummaryTool(),
    GetTopProductsTool.name: GetTopProductsTool(),
    GetInventoryStatusTool.name: GetInventoryStatusTool(),
    GetCampaignSpendTool.name: GetCampaignSpendTool(),
    GetSupportSentimentTool.name: GetSupportSentimentTool(),
    QueryMemoryTool.name: QueryMemoryTool(),
    SaveMemoryTool.name: SaveMemoryTool(),
}
