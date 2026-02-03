from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find the project root (where .env lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print("#" * 10)
print(f"Project root resolved to: {_PROJECT_ROOT}")
print("#" * 10)


class Settings(BaseSettings):
    app_name: str = Field(default="AI E-commerce Operations Brain", alias="APP_NAME")
    environment: str = Field(default="local", alias="APP_ENV")
    debug: bool = Field(default=False, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Database connections
    database_url: PostgresDsn | None = Field(default=None, alias="DATABASE_URL")
    vector_database_url: PostgresDsn | None = Field(default=None, alias="VECTOR_DATABASE_URL")

    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    @property
    def database_url_str(self) -> str | None:
        """Return database URL as string for SQLAlchemy."""
        if self.database_url is None:
            return None
        return str(self.database_url)

    @property
    def database_sync_url(self) -> str | None:
        """
        Alembic uses synchronous engines; convert +asyncpg driver to +psycopg automatically.
        """
        if self.database_url is None:
            return None
        return str(self.database_url).replace("+asyncpg", "+psycopg")

        # DIAL / Azure OpenAI

    dial_api_key: str | None = Field(default=None, alias="DIAL_API_KEY")
    dial_endpoint: AnyHttpUrl | None = Field(default=None, alias="DIAL_ENDPOINT")
    dial_deployment: str | None = Field(default=None, alias="DIAL_DEPLOYMENT")
    dial_api_version: str = Field(default="2023-12-01-preview", alias="DIAL_API_VERSION")
    dial_temperature: float = Field(default=0.2, alias="DIAL_TEMPERATURE")

    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="opsbrain-graph", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )
    langsmith_tracing_enabled: bool = Field(default=True, alias="LANGSMITH_TRACING_ENABLED")

    # MCP Endpoints
    mcp_server_port: int = Field(default=9001, alias="MCP_SERVER_PORT")
    mcp_sql_endpoint: AnyHttpUrl | None = Field(default=None, alias="MCP_SQL_ENDPOINT")
    mcp_api_key: str | None = Field(default=None, alias="MCP_API_KEY")

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def as_log_context(self) -> dict[str, Any]:
        """
        Convenience helper for structured logging (never log secrets).
        """
        return {
            "env": self.environment,
            "app_name": self.app_name,
            "debug": self.debug,
            "log_level": self.log_level,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
