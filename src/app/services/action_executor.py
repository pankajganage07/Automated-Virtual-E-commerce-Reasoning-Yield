"""
ActionExecutor: Executes approved HITL actions via MCP tools.

Maps action_type strings (stored in pending_actions table) to MCP tool names,
then invokes the tool with the stored payload.
"""

from __future__ import annotations

from typing import Any

from config import Settings
from opsbrain_graph.tools.mcp_client import MCPClient
from opsbrain_graph.tools.exceptions import MCPError, ToolInvocationError


# Mapping from action_type (what agents propose) to MCP tool name
ACTION_TYPE_TO_TOOL: dict[str, str] = {
    # Data Analyst - Custom SQL (HITL protected)
    "execute_custom_sql": "execute_sql_query",
    # Inventory actions
    "restock_item": "update_inventory",
    "update_inventory": "update_inventory",
    "adjust_stock": "update_inventory",
    # Marketing actions
    "pause_campaign": "update_campaign_status",
    "resume_campaign": "update_campaign_status",
    "update_campaign_status": "update_campaign_status",
    "adjust_budget": "update_campaign_budget",
    "update_campaign_budget": "update_campaign_budget",
    # Support actions
    "escalate_ticket": "escalate_ticket",
    "close_ticket": "close_ticket",
    "prioritize_ticket": "prioritize_ticket",
}


def transform_payload(action_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Transform agent action payload to MCP tool payload format.

    Different action types may use different field names than the MCP tools expect.
    This function normalizes the payload.
    """
    transformed = payload.copy()

    # restock_item uses "quantity" but update_inventory expects "quantity_change"
    if action_type == "restock_item":
        if "quantity" in transformed and "quantity_change" not in transformed:
            transformed["quantity_change"] = transformed.pop("quantity")
        if "reason" not in transformed:
            transformed["reason"] = "Restock requested by agent"

    # pause_campaign needs to set status="paused"
    elif action_type == "pause_campaign":
        transformed["status"] = "paused"
        if "reason" not in transformed:
            transformed["reason"] = "Campaign paused by agent recommendation"

    # resume_campaign needs to set status="active"
    elif action_type == "resume_campaign":
        transformed["status"] = "active"
        if "reason" not in transformed:
            transformed["reason"] = "Campaign resumed by agent recommendation"

    return transformed


class ActionExecutionError(Exception):
    """Raised when action execution fails."""

    def __init__(self, action_type: str, reason: str, details: Any = None):
        self.action_type = action_type
        self.reason = reason
        self.details = details
        super().__init__(f"Failed to execute action '{action_type}': {reason}")


class ActionExecutor:
    """
    Executes approved actions by calling the corresponding MCP tool.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mcp_base_url = f"http://localhost:{settings.mcp_server_port}"
        self._mcp_api_key = settings.mcp_api_key

    async def execute(
        self,
        action_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an action by invoking the mapped MCP tool.

        Args:
            action_type: The type of action (e.g., 'restock_item', 'pause_campaign')
            payload: The action payload to pass to the MCP tool

        Returns:
            The result from the MCP tool execution

        Raises:
            ActionExecutionError: If the action type is unknown or execution fails
        """
        tool_name = ACTION_TYPE_TO_TOOL.get(action_type)

        if tool_name is None:
            raise ActionExecutionError(
                action_type,
                f"Unknown action type. Valid types: {list(ACTION_TYPE_TO_TOOL.keys())}",
            )

        # Transform payload to match MCP tool expectations
        mcp_payload = transform_payload(action_type, payload)

        try:
            async with MCPClient(base_url=self._mcp_base_url, api_key=self._mcp_api_key) as client:
                result = await client.invoke(tool_name, mcp_payload)
                return {
                    "success": True,
                    "tool": tool_name,
                    "action_type": action_type,
                    "result": result,
                }
        except ToolInvocationError as exc:
            raise ActionExecutionError(
                action_type,
                f"MCP tool '{tool_name}' returned error: {exc}",
                details={"status_code": exc.status, "response": exc.message},
            ) from exc
        except MCPError as exc:
            raise ActionExecutionError(
                action_type,
                f"MCP communication error: {exc}",
            ) from exc
        except Exception as exc:
            raise ActionExecutionError(
                action_type,
                f"Unexpected error: {exc}",
            ) from exc

    def get_tool_for_action(self, action_type: str) -> str | None:
        """Get the MCP tool name for a given action type."""
        return ACTION_TYPE_TO_TOOL.get(action_type)

    def list_supported_actions(self) -> list[str]:
        """List all supported action types."""
        return list(ACTION_TYPE_TO_TOOL.keys())
