from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .mcp_client import MCPClient
from .exceptions import MCPError


class ExecuteSQLRequest(BaseModel):
    statement: str
    params: dict[str, Any] | None = None
    fetch: Literal["all", "one", "value"] = "all"


class ExecuteSQLResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    rowcount: int = 0
    columns: list[str] = Field(default_factory=list)


class SQLToolset:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def execute(self, payload: ExecuteSQLRequest) -> ExecuteSQLResponse:
        result = await self._client.invoke("execute_sql_query", payload.model_dump())
        try:
            return ExecuteSQLResponse.model_validate(result)
        except ValidationError as exc:
            raise MCPError(f"Invalid response for execute_sql_query: {exc}") from exc
