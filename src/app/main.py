from fastapi import FastAPI

from config import get_settings
from config.logging import setup_logging
from app.routers import actions, health, query

settings = get_settings()
setup_logging(level=settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    description="AI E-commerce Operations Brain API",
)

app.include_router(health.router)
app.include_router(actions.router)
app.include_router(query.router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {"message": "AI E-commerce Operations Brain is online."}
