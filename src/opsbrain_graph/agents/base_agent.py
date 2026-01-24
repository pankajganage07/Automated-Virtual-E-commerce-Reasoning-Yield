from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from config import Settings
from opsbrain_graph.tools import ToolRegistry


AgentStatus = Literal["success", "failure", "needs_retry"]


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

    def __init__(self, tools: ToolRegistry, settings: Settings) -> None:
        self.tools = tools
        self.settings = settings
        self.logger = logging.getLogger(f"agent.{self.name}")

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
