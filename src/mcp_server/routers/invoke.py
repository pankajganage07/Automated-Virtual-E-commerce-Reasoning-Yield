import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from mcp_server.config import MCPSettings, get_settings
from mcp_server.schemas import ErrorResponse, InvokeRequest, SuccessResponse, ToolMetadata
from mcp_server.tools.registry import TOOL_REGISTRY

logger = logging.getLogger("mcp.invoke")

router = APIRouter(tags=["Invoke"])


async def verify_api_key(request: Request, settings: MCPSettings = Depends(get_settings)) -> None:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    token = auth_header.split(" ", 1)[1]
    if token != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@router.post(
    "/invoke",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}},
)
async def invoke_tool(
    payload: InvokeRequest,
    _: None = Depends(verify_api_key),
) -> SuccessResponse:
    logger.info("=== MCP INVOKE === tool=%s, arguments=%s", payload.tool, payload.arguments)
    start = time.perf_counter()
    tool = TOOL_REGISTRY.get(payload.tool)

    if tool is None:
        logger.error("Unknown tool: %s", payload.tool)
        raise HTTPException(status_code=404, detail=f"Unknown tool '{payload.tool}'.")

    try:
        result = await tool(payload.arguments)
        logger.info("=== MCP INVOKE SUCCESS === tool=%s, result=%s", payload.tool, result)
    except ValueError as exc:
        logger.error("Tool %s ValueError: %s", payload.tool, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Tool %s Exception: %s", payload.tool, exc)
        raise HTTPException(status_code=500, detail=f"Tool '{payload.tool}' failed: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    metadata = ToolMetadata(tool=payload.tool, duration_ms=duration_ms)
    return SuccessResponse(result=result, metadata=metadata)
