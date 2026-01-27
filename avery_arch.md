# 🧠 Project Blueprint: AI E-commerce Operations Brain

## 1. Project Vision & Description

The **AI E-commerce Operations Brain** is an agentic system designed to function as an autonomous "Operations Manager." Unlike standard analytical tools that simply display data, this system **investigates**, **correlates**, **reasons**, and **acts**.

### Core Value Proposition

- **Cross-Domain Reasoning:** It doesn't just see a "sales drop"; it connects that drop to a stock-out event and a concurrent high-spend marketing campaign.
- **Historical Intuition:** It recalls past incidents to see if current strategies align with what worked previously.
- **Action with Accountability:** It can execute business logic (restocking, pausing ads) but maintains a strict **Human-in-the-Loop (HITL)** constraint for all state-changing actions.

---

## 2. Technical Stack

| **Layer** | **Technology** | **Rationale** |
| --- | --- | --- |
| **Orchestration** | **LangGraph** | Supports cyclic graphs, state persistence, and complex "wait-for-human" workflows. |
| **Data Protocol** | **MCP (Model Context Protocol)** | Standardizes how agents talk to the DB and external APIs, making tools "plug-and-play." |
| **Brain / LLM** | **GPT-4o or Claude 3.5 Sonnet** | High reasoning capability and superior tool-calling accuracy. |
| **Memory Store** | **PostgreSQL + pgvector** | Handles structured relational data and unstructured "lesson learned" embeddings. |
| **Backend API** | **FastAPI (Python)** | High-performance, asynchronous handling of agentic loops. |
| **Observability** | **LangSmith** | Essential for debugging multi-agent "reasoning traces" and cost monitoring. |
| **Evaluation** | **DeepEval** | Unit testing for agents to prevent regressions in logic or hallucinations. |

---

## 3. System Architecture & Approach

### The Supervisor-Worker Hybrid Design

We utilize a **top-down coordination** pattern. This prevents "Tool Explosion" (where one agent has 50 tools and gets confused).

1. **Supervisor (The Strategist):** Parses the user query, creates a "Battle Plan," and delegates tasks to specialized workers.
2. **Specialized Agents (The Subject Matter Experts):** Each agent has a narrow scope (Sales, Inventory, etc.) and a specific set of tools.
3. **The State Object:** A central, shared memory object in LangGraph that tracks findings, pending actions, and the "conversation history."

---

## 4. Database Schema (PostgreSQL)

### Operational Tables

- **`products`**: `id, name, category, price, stock_qty, low_stock_threshold`
- **`orders`**: `id, product_id, timestamp, qty, revenue, region, channel`
- **`campaigns`**: `id, name, budget, spend, clicks, conversions, status (active/paused)`
- **`support_tickets`**: `id, product_id, sentiment (0.0-1.0), issue_category, description`

### Agentic Intelligence Tables

- **`agent_memory`**: `id, embedding (vector), incident_summary, root_cause, action_taken, outcome`
- **`pending_actions`**: `id, agent_name, action_type, payload (JSON), reasoning, status (pending/approved/rejected)`

---

## 5. Agent Personas & Capabilities

| **Agent** | **Responsibility** | **Key Tools** |
| --- | --- | --- |
| **Supervisor** | Orchestration & Synthesis | `delegate_to_worker`, `final_response` |
| **Data Analyst** | Complex SQL & Joins | `execute_sql_query` (via MCP) |
| **Sales Agent** | Revenue & Trend Analysis | `get_revenue_metrics`, `identify_top_products` |
| **Inventory Agent** | Stock & Supply Chain | `check_stock`, `predict_stock_out`, `restock_item` (HITL) |
| **Marketing Agent** | Growth & Ad Performance | `get_ad_spend`, `calculate_roas`, `pause_campaign` (HITL) |
| **Support Agent** | Customer Friction Detection | `analyze_sentiment`, `get_ticket_trends` |
| **Historian** | RAG-based Memory Recall | `query_vector_memory`, `save_to_memory` |

---

## 6. Detailed Logic Flow (The "Lifecycle")

### Phase A: Diagnosis

1. **User Input:** "Why did sales drop 20% yesterday?"
2. **Supervisor Planning:** Determines it needs Sales data (to confirm the drop), Marketing data (to check ad spend), and Inventory data (to check availability).
3. **Parallel Execution:** Workers fetch their specific data points via MCP.
4. **Correlation:** The Sales Agent reports a drop in SKU #101. The Inventory Agent reports SKU #101 went out of stock at 2 PM. The Marketing Agent reports we spent $500 on ads for SKU #101 *after* it was out of stock.

### Phase B: Action & HITL

1. **Recommendation:** Supervisor suggests: "1. Restock SKU #101. 2. Pause the active campaign for SKU #101."
2. **Wait State:** The system writes these to `pending_actions` and sends a notification.
3. **Approval:** User clicks "Approve" in the UI.
4. **Execution:** The system triggers the `pause_campaign` and `restock_item` tools.

---

## 7. API Endpoints (FastAPI)

- `POST /query`: Primary endpoint for user questions. Returns the final synthesis.
- `GET /actions/pending`: Fetches all actions requiring human approval.
- `POST /actions/approve/{id}`: Triggers the actual tool execution for a specific action.
- `GET /history/incidents`: Retrieves past incidents and summaries from the vector DB.
- `GET /health`: Returns connectivity status of MCP servers and PostgreSQL.

---

## 8. Why This Approach?

- **Why LangGraph?** Traditional DAGs (Directed Acyclic Graphs) can't handle the "back-and-forth" required when an agent realizes it's missing data and needs to go back to a previous step.
- **Why MCP?** It isolates the database logic. If the store moves from PostgreSQL to Shopify API, we only update the MCP server, not the agent logic.
- **Why Vector Memory?** Business issues are often cyclical. Storing "Incident Reports" allows the AI to say, "This looks like the shipping delay we had last Christmas," providing immediate expert-level context.

---