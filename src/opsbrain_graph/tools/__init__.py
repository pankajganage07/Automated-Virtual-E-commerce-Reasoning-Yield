"""
LangGraph Agent Tools - Slimmed Architecture.

Each domain has 2 core tools. Complex queries route to DataAnalystAgent with HITL.
"""

from .registry import ToolRegistry

# SQL tools (for approved custom queries via HITL)
from .sql_tools import (
    SQLToolset,
    ExecuteSQLRequest,
    ExecuteSQLResponse,
    # Sales tools (2 core)
    SalesToolset,
    GetSalesSummaryRequest,
    SalesSummaryResponse,
    GetTopProductsRequest,
    GetTopProductsResponse,
    TopProduct,
)

# Inventory tools (2 core)
from .inventory_tools import (
    InventoryToolset,
    GetInventoryStatusRequest,
    GetInventoryStatusResponse,
    GetLowStockProductsRequest,
    GetLowStockProductsResponse,
    LowStockProduct,
    # Action tools (for HITL)
    UpdateInventoryRequest,
    UpdateInventoryResponse,
)

# Marketing tools (2 core)
from .marketing_tools import (
    MarketingToolset,
    GetCampaignSpendRequest,
    GetCampaignSpendResponse,
    CalculateROASRequest,
    CalculateROASResponse,
    CampaignROASInfo,
    # Action tools (for HITL)
    PauseCampaignRequest,
    PauseCampaignResponse,
    ResumeCampaignRequest,
    ResumeCampaignResponse,
    AdjustBudgetRequest,
    AdjustBudgetResponse,
)

# Support tools (2 core)
from .support_tools import (
    SupportToolset,
    GetSupportSentimentRequest,
    GetSupportSentimentResponse,
    GetTicketTrendsRequest,
    GetTicketTrendsResponse,
    TicketTrend,
    # Action tools (for HITL)
    EscalateTicketRequest,
    EscalateTicketResponse,
    CloseTicketRequest,
    CloseTicketResponse,
    PrioritizeTicketRequest,
    PrioritizeTicketResponse,
)

# Memory tools
from .memory_tools import (
    MemoryToolset,
    QueryVectorMemoryRequest,
    QueryVectorMemoryResponse,
    MemoryHit,
    SaveMemoryRequest,
    SaveMemoryResponse,
    ListIncidentsRequest,
    ListIncidentsResponse,
    IncidentRecord,
)

__all__ = [
    "ToolRegistry",
    # SQL tools
    "SQLToolset",
    "ExecuteSQLRequest",
    "ExecuteSQLResponse",
    # Sales tools (2 core)
    "SalesToolset",
    "GetSalesSummaryRequest",
    "SalesSummaryResponse",
    "GetTopProductsRequest",
    "GetTopProductsResponse",
    "TopProduct",
    # Inventory tools (2 core)
    "InventoryToolset",
    "GetInventoryStatusRequest",
    "GetInventoryStatusResponse",
    "GetLowStockProductsRequest",
    "GetLowStockProductsResponse",
    "LowStockProduct",
    "UpdateInventoryRequest",
    "UpdateInventoryResponse",
    # Marketing tools (2 core)
    "MarketingToolset",
    "GetCampaignSpendRequest",
    "GetCampaignSpendResponse",
    "CalculateROASRequest",
    "CalculateROASResponse",
    "CampaignROASInfo",
    "PauseCampaignRequest",
    "PauseCampaignResponse",
    "ResumeCampaignRequest",
    "ResumeCampaignResponse",
    "AdjustBudgetRequest",
    "AdjustBudgetResponse",
    # Support tools (2 core)
    "SupportToolset",
    "GetSupportSentimentRequest",
    "GetSupportSentimentResponse",
    "GetTicketTrendsRequest",
    "GetTicketTrendsResponse",
    "TicketTrend",
    "EscalateTicketRequest",
    "EscalateTicketResponse",
    "CloseTicketRequest",
    "CloseTicketResponse",
    "PrioritizeTicketRequest",
    "PrioritizeTicketResponse",
    # Memory tools
    "MemoryToolset",
    "QueryVectorMemoryRequest",
    "QueryVectorMemoryResponse",
    "MemoryHit",
    "SaveMemoryRequest",
    "SaveMemoryResponse",
    "ListIncidentsRequest",
    "ListIncidentsResponse",
    "IncidentRecord",
]
