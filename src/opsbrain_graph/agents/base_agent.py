from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TYPE_CHECKING

from config import Settings
from opsbrain_graph.tools import ToolRegistry

if TYPE_CHECKING:
    from opsbrain_graph.memory import MemoryService

AgentStatus = Literal["success", "failure", "needs_retry"]


# =============================================================================
# Agent Metadata - Used for LLM-based task planning
# =============================================================================


@dataclass
class AgentCapability:
    """Describes a single capability/mode of an agent."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # param_name -> description
    example_queries: list[str] = field(default_factory=list)


@dataclass
class AgentMetadata:
    """Rich metadata for LLM planning prompt generation."""

    name: str
    display_name: str
    description: str
    capabilities: list[AgentCapability] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # Trigger keywords for fallback
    priority_boost: list[str] = field(default_factory=list)  # Keywords that boost priority

    def to_prompt_section(self) -> str:
        """Generate a prompt section describing this agent."""
        lines = [f"### {self.display_name} Agent"]
        lines.append(f"- {self.description}")

        if self.capabilities:
            lines.append("- Capabilities:")
            for cap in self.capabilities:
                params_str = ""
                if cap.parameters:
                    params_str = f" (parameters: {', '.join(cap.parameters.keys())})"
                lines.append(f"  * {cap.name}: {cap.description}{params_str}")

        if self.keywords:
            lines.append(f"- Trigger keywords: {', '.join(self.keywords)}")

        return "\n".join(lines)


@dataclass
class AgentRecommendation:
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    requires_approval: bool = True  # default HITL


@dataclass
class AgentTask:
    agent: str
    objective: str
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: Literal["low", "normal", "high"] = "normal"
    result_slot: str | None = None  # e.g. "agent_findings.sales"


@dataclass
class AgentRunContext:
    user_query: str
    conversation_history: Sequence[dict[str, Any]] = field(default_factory=list)
    memory_context: Sequence[dict[str, Any]] = field(default_factory=list)
    state_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    status: AgentStatus
    findings: dict[str, Any] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    recommendations: list[AgentRecommendation] = field(default_factory=list)
    errors: str | None = None


class BaseAgent(abc.ABC):
    name: str = "base"
    description: str = "Base agent"
    metadata: AgentMetadata | None = None  # Override in subclasses

    def __init__(
        self, tools: ToolRegistry, settings: Settings, memory_service: "MemoryService | None" = None
    ) -> None:
        self.tools = tools
        self.settings = settings
        self.memory_service = memory_service
        self.logger = logging.getLogger(f"agent.{self.name}")

    @classmethod
    def get_metadata(cls) -> AgentMetadata:
        """Get agent metadata for LLM planning. Override in subclasses."""
        if cls.metadata:
            return cls.metadata
        # Default metadata from class attributes
        return AgentMetadata(
            name=cls.name,
            display_name=cls.name.upper(),
            description=cls.description,
        )

    @abc.abstractmethod
    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        raise NotImplementedError

    def success(
        self,
        findings: dict[str, Any] | None = None,
        insights: list[str] | None = None,
        recommendations: list[AgentRecommendation] | None = None,
    ) -> AgentResult:
        return AgentResult(
            status="success",
            findings=findings or {},
            insights=insights or [],
            recommendations=recommendations or [],
        )

    def failure(self, error: Exception | str, needs_retry: bool = False) -> AgentResult:
        message = str(error)
        self.logger.exception("%s agent failed: %s", self.name, message)
        status: AgentStatus = "needs_retry" if needs_retry else "failure"
        return AgentResult(status=status, errors=message)
