from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings
from .exceptions import MCPError, ToolInvocationError


class MCPClient:
    """
    Lightweight async client for MCP servers.
    MCP servers are assumed to expose POST /invoke with payload:
        {"tool": "<tool_name>", "arguments": {...}}
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def invoke(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = {"tool": tool_name, "arguments": arguments or {}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._client.post("/invoke", json=payload, headers=headers)

        if response.status_code >= 400:
            raise ToolInvocationError(tool_name, response.status_code, response.text)

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise MCPError(f"Invalid JSON response from MCP tool {tool_name}: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
