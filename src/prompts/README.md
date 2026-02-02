# Prompts Directory

This directory contains all LLM prompts used by the OpsBrain system.

## File Format

Prompts are stored in YAML format for:
- Easy editing with syntax highlighting
- Support for multi-line strings
- Metadata (version, description, author)
- Variable templating with Jinja2 syntax

## Files

| File | Purpose |
|------|---------|
| `planning.yaml` | Supervisor planning prompts (task assignment) |
| `synthesis.yaml` | Result synthesis and answer generation |
| `sql_generation.yaml` | SQL generation from natural language |
| `db_schema.yaml` | Database schema context for SQL generation |

## Template Variables

Prompts support Jinja2 templating. Common variables:
- `{{ agent_capabilities }}` - Dynamic agent list
- `{{ db_schema }}` - Database schema context
- `{{ query }}` - User's question

## Usage

```python
from prompts import get_prompt, render_prompt

# Get a prompt by key
prompt = get_prompt("planning.system")

# Render with variables
rendered = render_prompt("sql_generation.generate", query="What are top products?")
```
