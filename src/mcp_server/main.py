from fastapi import FastAPI

from .config import get_settings
from .logging import setup_logging
from mcp_server.routers import health, invoke

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="MCP Server bridging tools to the operational database.",
)

app.include_router(health.router)
app.include_router(invoke.router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {"message": "MCP server ready."}
