class MCPError(Exception):
    """Base class for MCP-related errors."""


class ToolInvocationError(MCPError):
    """Raised when the MCP server fails to execute the requested tool."""

    def __init__(self, tool_name: str, status: int, message: str) -> None:
        super().__init__(f"[{tool_name}] status={status} message={message}")
        self.tool_name = tool_name
        self.status = status
        self.message = message
