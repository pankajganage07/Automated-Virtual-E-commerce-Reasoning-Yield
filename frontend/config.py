"""
Frontend configuration settings.
"""

from dataclasses import dataclass
import os


@dataclass
class FrontendConfig:
    """Configuration for the Streamlit frontend."""

    # Backend API settings
    api_base_url: str = os.getenv("AVERY_API_URL", "http://localhost:8000")
    api_timeout: int = 120  # seconds - LLM calls can be slow

    # UI settings
    page_title: str = "Avery OpsBrain"
    page_icon: str = "🧠"

    # Chat settings
    max_history_messages: int = 50  # Keep last N messages in session

    # Polling settings (for checking pending actions)
    poll_interval: int = 5  # seconds


def get_config() -> FrontendConfig:
    """Get frontend configuration."""
    return FrontendConfig()
