from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from .mcp_client import MCPClient
from .sql_tools import SQLToolset
from .inventory_tools import InventoryToolset
from .marketing_tools import MarketingToolset
from .support_tools import SupportToolset
from .memory_tools import MemoryToolset


@dataclass
class ToolRegistry:
    sql: SQLToolset
    inventory: InventoryToolset
    marketing: MarketingToolset
    support: SupportToolset
    memory: MemoryToolset

    @classmethod
    def from_settings(cls, settings: Settings) -> "ToolRegistry":
        sql_client = MCPClient(settings.mcp_sql_endpoint)
        inventory_client = MCPClient(settings.mcp_inventory_endpoint)
        marketing_client = MCPClient(settings.mcp_marketing_endpoint)
        support_client = MCPClient(settings.mcp_support_endpoint)
        memory_client = MCPClient(settings.mcp_memory_endpoint)

        return cls(
            sql=SQLToolset(sql_client),
            inventory=InventoryToolset(inventory_client),
            marketing=MarketingToolset(marketing_client),
            support=SupportToolset(support_client),
            memory=MemoryToolset(memory_client),
        )
