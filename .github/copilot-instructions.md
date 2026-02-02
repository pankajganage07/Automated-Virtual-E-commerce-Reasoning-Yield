# Avery (AI E-commerce Operations Brain) - Copilot Instructions

## Architecture Overview

This is an **agentic AI system** using a Supervisor-Worker pattern for autonomous e-commerce operations. Key components:

- **FastAPI Backend** (`src/app/`) - Main API on port 8000, routes user queries to the LangGraph orchestrator
- **MCP Server** (`src/mcp_server/`) - Tool abstraction layer on port 9001, isolates database access from agent logic
- **LangGraph Orchestrator** (`src/opsbrain_graph/`) - Manages agent workflows with Human-in-the-Loop (HITL) state persistence
- **Streamlit Frontend** (`frontend/`) - Chat interface with HITL approval workflow

Data flows: `Frontend → FastAPI → OrchestratorService → LangGraph → Agents → MCP Client → MCP Server → PostgreSQL`

## Project Structure

```
src/
├── app/routers/query.py      # Main /query endpoint - entry point for all queries
├── opsbrain_graph/
│   ├── supervisor.py         # LLM-based task planning and synthesis
│   ├── graph.py              # LangGraph state machine with HITL checkpointing
│   ├── agents/               # Specialized agents (sales, inventory, marketing, support, historian, data_analyst)
│   └── tools/                # MCP client toolsets - agents never call DB directly
├── mcp_server/
│   ├── routers/invoke.py     # POST /invoke endpoint for tool execution
│   └── tools/                # Actual DB query implementations
└── prompts/                  # YAML-based prompts with Jinja2 templating
```

## Key Patterns

### Adding a New Agent
1. Create `src/opsbrain_graph/agents/<name>_agent.py` extending `BaseAgent`
2. Define `AgentMetadata` with capabilities, keywords, and example queries - this auto-generates planning prompts
3. Register in `graph.py` `_agents` dict
4. Add corresponding MCP tools in `src/mcp_server/tools/` if new DB access needed

### Tool Invocation Pattern
Agents **never access the database directly**. All data access goes through MCP:
```python
# In agent code - use self.tools.<toolset>.<method>()
result = await self.tools.sales.get_summary(window_days=7)
# This calls MCP Server's /invoke endpoint with tool="get_sales_summary"
```

### Prompts Management
Prompts live in `src/prompts/*.yaml`. Access via:
```python
from prompts import get_prompt, render_prompt
prompt = get_prompt("planning.system_static.template")
rendered = render_prompt("synthesis.system", findings=data)
```

### HITL (Human-in-the-Loop) Actions
State-changing actions require approval. Use `AgentRecommendation` with `requires_approval=True`:
```python
AgentRecommendation(
    action_type="pause_campaign",
    payload={"campaign_id": 123},
    reasoning="Spending on out-of-stock product",
    requires_approval=True
)
```

## Development Commands

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Run main API (port 8000)
uvicorn src.app.main:app --reload

# Run MCP Server (port 9001) - required for agents to work
uvicorn src.mcp_server.main:app --host 0.0.0.0 --port 9001 --reload

# Run Streamlit frontend
streamlit run frontend/app.py

# Database migrations
alembic upgrade head
```

## Configuration

All settings via `.env` file (see `src/config/settings.py` for full list):
- `DATABASE_URL` - PostgreSQL with asyncpg driver
- `DIAL_API_KEY`, `DIAL_ENDPOINT`, `DIAL_DEPLOYMENT` - Azure OpenAI/DIAL for LLM
- `MCP_SQL_ENDPOINT`, `MCP_API_KEY` - MCP server connection
- `LANGSMITH_API_KEY` - Optional tracing

## Database Schema

Core tables in `src/db/models.py`: `products`, `orders`, `campaigns`, `support_tickets`
Agentic tables: `agent_memory` (pgvector for RAG), `pending_actions` (HITL queue)

## Code Style

- Python 3.11+, async/await throughout
- Pydantic v2 for all schemas and validation
- Type hints required (`Mapped[]` for SQLAlchemy models)
- Line length: 100 (ruff/black)
- Imports: standard lib → third-party → local (`from config import ...`)
