from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from config import Settings, get_settings


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


@lru_cache
def _create_llm() -> AzureChatOpenAI:
    """Cached LLM creation - no parameters needed."""
    settings = get_settings()
    _validate_settings(settings)

    return AzureChatOpenAI(
        azure_endpoint=str(settings.dial_endpoint),
        azure_deployment=settings.dial_deployment,
        api_key=settings.dial_api_key,
        api_version=settings.dial_api_version,
        temperature=settings.dial_temperature,
        streaming=False,
    )


def get_llm() -> AzureChatOpenAI:
    """Get singleton LLM instance."""
    return _create_llm()
