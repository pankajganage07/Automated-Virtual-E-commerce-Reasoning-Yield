from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    app_name: str = Field(default="MCP Database Server", alias="MCP_APP_NAME")
    host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    port: int = Field(default=9001, alias="MCP_PORT")

    database_url: str = Field(..., alias="MCP_DB_URL")
    api_key: str = Field(..., alias="MCP_API_KEY")

    log_level: str = Field(default="INFO", alias="MCP_LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> MCPSettings:
    return MCPSettings()
