from functools import lru_cache
import logging

from langchain_openai import AzureChatOpenAI

from config import Settings, get_settings
from utils.observability import LangSmithConfig

logger = logging.getLogger("opsbrain.llm")


def _validate_settings(settings: Settings) -> None:
    required = {
        "dial_api_key": settings.dial_api_key,
        "dial_endpoint": settings.dial_endpoint,
        "dial_deployment": settings.dial_deployment,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"DIAL LLM configuration incomplete, missing: {', '.join(missing)}. "
            "Set them in the environment (.env)."
        )


def _initialize_langsmith(settings: Settings) -> None:
    """Initialize LangSmith tracing if configured."""
    if not settings.langsmith_tracing_enabled:
        logger.info("LangSmith tracing disabled via configuration")
        return

    LangSmithConfig.set_project(settings.langsmith_project)
    enabled = LangSmithConfig.initialize(settings)

    if enabled:
        logger.info(
            "LangSmith tracing enabled for LLM calls (project: %s)",
            settings.langsmith_project,
        )


@lru_cache
def _create_llm() -> AzureChatOpenAI:
    """Cached LLM creation - no parameters needed."""
    settings = get_settings()
    _validate_settings(settings)

    # Initialize LangSmith before creating LLM
    # This ensures all LLM calls are traced automatically
    _initialize_langsmith(settings)

    return AzureChatOpenAI(
        azure_endpoint=str(settings.dial_endpoint),
        azure_deployment=settings.dial_deployment,
        api_key=settings.dial_api_key,
        api_version=settings.dial_api_version,
        temperature=settings.dial_temperature,
        streaming=False,
    )


def get_llm() -> AzureChatOpenAI:
    """Get singleton LLM instance with LangSmith tracing enabled."""
    return _create_llm()
