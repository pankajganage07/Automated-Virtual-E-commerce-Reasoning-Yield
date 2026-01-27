## 🏗 MCP Architecture Overview

### Purpose

The MCP Server acts as a standardized, secure abstraction layer between the AI Ops Brain (agents, LangGraph workflows) and the operational PostgreSQL database.

### Mission

- Provide tool-like actions via the MCP protocol, accessible through a simple `/invoke` API.
- Centralize data access and enforce validation, authentication, observability, and extensibility.
- Allow agents (Sales, Inventory, Marketing, Support, Historian, Data Analyst) to fetch real data without direct DB connections.

---

## 🔄 High-Level Flow

```
FastAPI (AI Ops Brain) → Orchestrator → LangGraph → Agents
 → Tool Registry → MCP Client → MCP Server (tools executed)
 → PostgreSQL (operational DB, vector memory)

```

---

## 🔌 MCP Server Responsibilities

1. **Standard Tool Interface**
    
    Exposes a single endpoint (`POST /invoke`) where agents call tools by name, passing validated arguments.
    
2. **Security**
    
    Uses API key-based authentication (`Authorization: Bearer <MCP_API_KEY>`). Optional: role-based access or mTLS later.
    
3. **Validation & Policies**
    
    Each tool defines expected arguments via Pydantic models. Rejects invalid input and can enforce guardrails (e.g., prevent unbounded SQL queries).
    
4. **Observability**
    
    Logs each tool invocation, records duration, and can emit metrics around tool usage and database impact.
    
5. **Extensibility**
    
    Tools are implemented as modular classes; adding a new operation (e.g., `get_supplier_delays`) only requires a new tool, no changes to agents.
    

---

## 🧱 Component Breakdown

### 1. MCP Server (FastAPI-based)

- **Routing**: `/invoke` endpoint routes to the requested tool.
- **Authentication**: Verifies API key header before processing requests.
- **Error Handling**: Returns structured responses indicating success/failure with detailed metadata.
- **Configuration**: Uses `.env` (e.g., `MCP_DB_URL`, `MCP_API_KEY`, `MCP_HOST`, `MCP_PORT`).

### 2. Tool Registry

- Maps tool names to implementations (e.g., `get_sales_summary`, `execute_sql_query`, etc.).
- Tools define their own request schema and async DB access logic.
- Tools use a shared async SQLAlchemy session (`mcp_server/db.py`), pointing to the main ops DB.

### 3. Core Tools Implemented

**General:**

- `execute_sql_query`: Parameterized SQL read execution (fetch modes: rows/one/value).

**Sales:**

- `get_sales_summary`: Aggregates revenue/units over past N days grouped by day/week.
- `get_top_products`: Top products by revenue or units.

**Inventory:**

- `get_inventory_status`: Fetches stock level and thresholds for product IDs.

**Marketing:**

- `get_campaign_spend`: Current spend, clicks, conversions, status per campaign.

**Support:**

- `get_support_sentiment`: Count, average sentiment, negative ratio over time window.

**Memory:**

- `query_vector_memory`: Retrieves similar incident summaries via pgvector.
- `save_to_memory`: Persists new incident with embedding.

*(Additional domain tools can be registered similarly.)*

### 4. Database Layer

- MCP has its own SQLAlchemy engine and async sessions but accesses the same Postgres DB.
- Reuses existing tables (`orders`, `products`, `campaigns`, `support_tickets`, `agent_memory`, etc.).
- Embedding operations leverage `pgvector`.

---

## ⚙️ Tool Invocation Contract

### Request

```json
POST /invoke
Authorization: Bearer <MCP_API_KEY>
Content-Type: application/json

{
  "tool": "get_sales_summary",
  "arguments": {
    "window_days": 7,
    "group_by": "day"
  }
}

```

### Success Response

```json
{
  "success": true,
  "result": {
    "summary": { "total_revenue": 12345.67, "total_units": 512 },
    "trend": [{ "bucket": "2024-06-07", "revenue": 2356.78, "units": 90 }]
  },
  "metadata": {
    "tool": "get_sales_summary",
    "duration_ms": 38.5
  }
}

```

### Error Response

```json
{
  "success": false,
  "error": {
    "type": "ValidationError",
    "message": "window_days must be >= 1"
  }
}

```

---

## 🔐 Security

- API key required on every request.
- MCP server validates the key before invoking the tool.
- All agent instances share this key (stored in config/environment).
- DB credentials remain only inside the MCP server.

---

## 📡 Deployment

- Runs as a standalone FastAPI/Uvicorn service (e.g., `uvicorn mcp_server.main:app --port 9001`), optionally containerized.
- Can be horizontally scaled (stateless, except DB access).
- Logs can be centralized or forwarded for observability.

---

## 🧩 Integration with AI Ops Brain

1. `ToolRegistry` in LangGraph application uses `MCPClient` to call MCP server.
2. Each agent (Sales, Inventory, etc.) uses typed tool wrappers (e.g., `GetSalesSummaryRequest`) that internally call `MCPClient.invoke`.
3. Supervisor/Graph remains unchanged—tools are simply remote calls now.

---

## ✅ Summary

This architecture provides clear separation between AI agent logic and the underlying data systems, offering secure, consistent, and maintainable access to operational data. All tool calls go through the MCP layer, ensuring observability and future extensibility.

Feel free to hand this description to any LLM or team to bootstrap further development or enhancements!