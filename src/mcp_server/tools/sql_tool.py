from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


class ExecuteSQLPayload(BaseModel):
    statement: str
    params: dict[str, Any] | None = None
    fetch: Literal["all", "one", "value"] = "all"


class ExecuteSQLTool(BaseTool):
    name = "execute_sql_query"

    def request_model(self) -> type[BaseModel]:
        return ExecuteSQLPayload

    async def run(self, session, payload: ExecuteSQLPayload) -> dict[str, Any]:
        result = await session.execute(text(payload.statement), payload.params or {})

        if payload.fetch == "value":
            row = result.scalar()
            return {"value": row}

        rows = [dict(row._mapping) for row in result]
        if payload.fetch == "one":
            return {"row": rows[0] if rows else None}

        return {
            "rows": rows,
            "rowcount": result.rowcount or len(rows),
            "columns": list(rows[0].keys()) if rows else [],
        }
