"""
LangSmith observability integration for multi-agent reasoning traces and cost monitoring.

This module provides:
- LangSmith tracing configuration
- Context managers for run tracking
- Callback handlers for LangGraph and LangChain
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from langsmith import Client
from langsmith.run_helpers import traceable

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger("opsbrain.observability")


class LangSmithConfig:
    """Configuration and state for LangSmith integration."""

    _initialized: bool = False
    _client: Client | None = None
    _project_name: str = "opsbrain-graph"
    _tracing_enabled: bool = False

    @classmethod
    def initialize(cls, settings: "Settings") -> bool:
        """
        Initialize LangSmith with configuration from settings.

        Args:
            settings: Application settings containing langsmith_api_key

        Returns:
            True if LangSmith was successfully initialized
        """
        if cls._initialized:
            return cls._tracing_enabled

        # Check if tracing is disabled via settings
        if (
            hasattr(settings, "langsmith_tracing_enabled")
            and not settings.langsmith_tracing_enabled
        ):
            logger.info("LangSmith tracing disabled via LANGSMITH_TRACING_ENABLED=false")
            cls._initialized = True
            cls._tracing_enabled = False
            return False

        api_key = settings.langsmith_api_key
        if not api_key or api_key == "ls-...":
            logger.info("LangSmith API key not configured, tracing disabled")
            cls._initialized = True
            cls._tracing_enabled = False
            return False

        try:
            # Get project name from settings if available
            project_name = getattr(settings, "langsmith_project", cls._project_name)
            cls._project_name = project_name

            # Get endpoint from settings if available
            endpoint = getattr(settings, "langsmith_endpoint", "https://api.smith.langchain.com")

            # Set environment variables for LangChain/LangGraph auto-tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = project_name
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint

            # Create client for programmatic access
            cls._client = Client(api_key=api_key, api_url=endpoint)

            cls._initialized = True
            cls._tracing_enabled = True

            logger.info(
                "LangSmith tracing enabled (project: %s, env: %s)",
                project_name,
                settings.environment,
            )
            return True

        except Exception as exc:
            logger.warning("Failed to initialize LangSmith: %s", exc)
            cls._initialized = True
            cls._tracing_enabled = False
            return False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if LangSmith tracing is enabled."""
        return cls._tracing_enabled

    @classmethod
    def get_client(cls) -> Client | None:
        """Get the LangSmith client instance."""
        return cls._client

    @classmethod
    def set_project(cls, project_name: str) -> None:
        """Set the LangSmith project name."""
        cls._project_name = project_name
        if cls._tracing_enabled:
            os.environ["LANGCHAIN_PROJECT"] = project_name


@lru_cache
def get_langsmith_client(settings: "Settings") -> Client | None:
    """Get or create LangSmith client (cached)."""
    LangSmithConfig.initialize(settings)
    return LangSmithConfig.get_client()


def trace_agent(
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator for tracing agent executions.

    Usage:
        @trace_agent("sales_agent")
        async def run(self, task, context):
            ...

    Args:
        name: Name for the trace (defaults to function name)
        run_type: LangSmith run type (chain, llm, tool, etc.)
        metadata: Additional metadata to include in trace
    """

    def decorator(func: Callable) -> Callable:
        if not LangSmithConfig.is_enabled():
            return func

        return traceable(
            name=name,
            run_type=run_type,
            metadata=metadata or {},
        )(func)

    return decorator


def trace_tool(
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator for tracing tool executions.

    Usage:
        @trace_tool("get_orders")
        async def get_orders(self, params):
            ...
    """
    return trace_agent(name=name, run_type="tool", metadata=metadata)


@contextmanager
def trace_span(
    name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
):
    """
    Context manager for creating trace spans.

    Usage:
        with trace_span("planning", inputs={"query": query}):
            tasks = await supervisor.plan(state)
    """
    if not LangSmithConfig.is_enabled():
        yield
        return

    # For sync context, we use basic logging
    # LangSmith auto-traces LangChain/LangGraph operations
    logger.debug("Trace span: %s (%s)", name, run_type)
    try:
        yield
    finally:
        logger.debug("End trace span: %s", name)


@asynccontextmanager
async def async_trace_span(
    name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
):
    """
    Async context manager for creating trace spans.

    Usage:
        async with async_trace_span("agent_execution", inputs={"agent": "sales"}):
            result = await agent.run(task, context)
    """
    if not LangSmithConfig.is_enabled():
        yield
        return

    logger.debug("Async trace span: %s (%s)", name, run_type)
    try:
        yield
    finally:
        logger.debug("End async trace span: %s", name)


class TracingCallbackHandler:
    """
    Callback handler for custom trace events.

    Use this to add custom events to LangSmith traces,
    such as agent decisions, tool selections, or HITL events.
    """

    @staticmethod
    def on_agent_start(agent_name: str, task: Any, context: Any) -> None:
        """Log agent start event."""
        if not LangSmithConfig.is_enabled():
            return
        logger.info("[Trace] Agent started: %s", agent_name)

    @staticmethod
    def on_agent_end(agent_name: str, result: Any, duration_ms: float) -> None:
        """Log agent completion event."""
        if not LangSmithConfig.is_enabled():
            return
        logger.info(
            "[Trace] Agent completed: %s (status: %s, duration: %.2fms)",
            agent_name,
            getattr(result, "status", "unknown"),
            duration_ms,
        )

    @staticmethod
    def on_tool_call(tool_name: str, params: dict[str, Any]) -> None:
        """Log tool invocation event."""
        if not LangSmithConfig.is_enabled():
            return
        logger.info("[Trace] Tool called: %s", tool_name)

    @staticmethod
    def on_hitl_gate(thread_id: str, pending_actions: int) -> None:
        """Log HITL gate event."""
        if not LangSmithConfig.is_enabled():
            return
        logger.info(
            "[Trace] HITL gate: thread=%s, pending_actions=%d",
            thread_id,
            pending_actions,
        )

    @staticmethod
    def on_hitl_resume(thread_id: str, approved: int, rejected: int) -> None:
        """Log HITL resume event."""
        if not LangSmithConfig.is_enabled():
            return
        logger.info(
            "[Trace] HITL resume: thread=%s, approved=%d, rejected=%d",
            thread_id,
            approved,
            rejected,
        )


def get_run_url(run_id: str) -> str | None:
    """
    Get the LangSmith URL for a specific run.

    Args:
        run_id: The LangSmith run ID

    Returns:
        URL to view the run in LangSmith, or None if not available
    """
    client = LangSmithConfig.get_client()
    if not client:
        return None

    try:
        run = client.read_run(run_id)
        return run.url if hasattr(run, "url") else None
    except Exception:
        return None


def log_feedback(
    run_id: str,
    key: str,
    score: float | None = None,
    value: str | None = None,
    comment: str | None = None,
) -> bool:
    """
    Log feedback for a LangSmith run.

    Useful for recording user feedback on agent responses.

    Args:
        run_id: The LangSmith run ID
        key: Feedback key (e.g., "correctness", "helpfulness")
        score: Numeric score (0-1)
        value: Categorical value
        comment: Text comment

    Returns:
        True if feedback was logged successfully
    """
    client = LangSmithConfig.get_client()
    if not client:
        return False

    try:
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            value=value,
            comment=comment,
        )
        logger.info("Logged feedback for run %s: %s", run_id, key)
        return True
    except Exception as exc:
        logger.warning("Failed to log feedback: %s", exc)
        return False
