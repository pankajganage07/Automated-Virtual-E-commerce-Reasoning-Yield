from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opsbrain_graph.agents.base_agent import AgentRecommendation, AgentTask


@dataclass
class PendingActionProposal:
    agent_name: str
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    requires_approval: bool = True


@dataclass
class DiagnosisSummary:
    narrative: str = ""
    key_findings: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class GraphState:
    user_query: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    battle_plan: list[AgentTask] = field(default_factory=list)
    agent_findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_insights: dict[str, list[str]] = field(default_factory=dict)

    structured_evidence: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: DiagnosisSummary = field(default_factory=DiagnosisSummary)

    recommendations: list[AgentRecommendation] = field(default_factory=list)
    pending_action_proposals: list[PendingActionProposal] = field(default_factory=list)

    memory_context: list[dict[str, Any]] = field(default_factory=list)
    system_warnings: list[str] = field(default_factory=list)

    hitl_wait: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_agent_result(
        self,
        agent_name: str,
        findings: dict[str, Any],
        insights: list[str],
        recommendations: list[AgentRecommendation],
    ) -> None:

        if agent_name == "historian" and "matches" in findings:
            self.memory_context = findings["matches"]

        self.agent_findings[agent_name] = findings
        self.agent_insights[agent_name] = insights
        self.recommendations.extend(recommendations)

    def add_warning(self, message: str) -> None:
        self.system_warnings.append(message)
