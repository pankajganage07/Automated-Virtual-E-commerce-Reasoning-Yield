from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from .mcp_client import MCPClient
from .sql_tools import SQLToolset, SalesToolset
from .inventory_tools import InventoryToolset
from .marketing_tools import MarketingToolset
from .support_tools import SupportToolset
from .memory_tools import MemoryToolset


@dataclass
class ToolRegistry:
    sql: SQLToolset
    sales: SalesToolset
    inventory: InventoryToolset
    marketing: MarketingToolset
    support: SupportToolset
    memory: MemoryToolset

    @classmethod
    def from_settings(cls, settings: Settings) -> "ToolRegistry":
        client = MCPClient(
            base_url=settings.mcp_sql_endpoint,
            api_key=settings.mcp_api_key,
        )
        # Reuse same MCP client for all toolsets (since MCP wraps all tools)
        return cls(
            sql=SQLToolset(client),
            sales=SalesToolset(client),
            inventory=InventoryToolset(client),
            marketing=MarketingToolset(client),
            support=SupportToolset(client),
            memory=MemoryToolset(client),
        )
