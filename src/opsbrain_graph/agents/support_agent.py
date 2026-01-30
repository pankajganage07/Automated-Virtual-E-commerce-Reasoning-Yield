"""
Support Agent - Slimmed architecture (2 core capabilities).

Capabilities:
1. sentiment_analysis - Get support sentiment metrics
2. ticket_trends - Analyze ticket trends by category/product

Complex queries (common issues, complaint comparison) route to DataAnalystAgent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opsbrain_graph.tools import GetSupportSentimentRequest, GetTicketTrendsRequest
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentResult,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)

logger = logging.getLogger("agent.support")


# Query patterns that this agent CANNOT handle (require DataAnalystAgent)
COMPLEX_QUERY_PATTERNS = [
    r"common.*issue",
    r"frequent.*problem",
    r"top.*complaint",
    r"most.*reported",
    r"compare.*complaint",
    r"complaint.*increase",
    r"complaint.*decrease",
    r"yesterday.*vs.*week",
    r"today.*vs.*last",
    r"complaint.*trend.*comparison",
    r"issue.*spike",
    r"issue.*drop",
    r"product.*support.*correlation",
    r"support.*by.*region",
    r"resolution.*time",
    r"agent.*performance",
]


class SupportAgent(BaseAgent):
    """
    Support Agent with 2 core capabilities.

    Complex queries trigger cannot_handle for routing to DataAnalystAgent.
    """

    name = "support"
    description = "Analyzes support sentiment and ticket trends."

    metadata = AgentMetadata(
        name="support",
        display_name="SUPPORT",
        description="Analyzes support sentiment and ticket trends. For complex analytics (common issues, period comparison), use data analyst.",
        capabilities=[
            AgentCapability(
                name="sentiment_analysis",
                description="Analyze overall support sentiment and negative ticket ratios",
                parameters={
                    "window_days": "Time period to analyze (default: 7)",
                    "product_id": "Optional product ID to filter by",
                },
                example_queries=[
                    "What's the customer sentiment like?",
                    "Are there many complaints?",
                    "How is support trending?",
                ],
            ),
            AgentCapability(
                name="ticket_trends",
                description="Analyze ticket trends by category, product, or day",
                parameters={
                    "window_days": "Analysis window (default: 14)",
                    "group_by": "Group by: issue_category, product, day",
                    "product_id": "Optional product filter",
                },
                example_queries=[
                    "What are the ticket trends?",
                    "Show me support volume by category",
                    "Ticket breakdown by product",
                ],
            ),
        ],
        keywords=[
            "ticket",
            "support",
            "sentiment",
            "complaint",
            "customer",
            "issue",
            "feedback",
        ],
        priority_boost=["angry customers", "high complaints", "negative sentiment"],
    )

    def _is_complex_query(self, query: str) -> bool:
        """Check if query requires complex analysis."""
        query_lower = query.lower()
        for pattern in COMPLEX_QUERY_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _cannot_handle(self, query: str) -> AgentResult:
        """Return cannot_handle status for supervisor to route to analyst."""
        return AgentResult(
            status="cannot_handle",
            findings={
                "query": query,
                "reason": "This query requires complex support analysis (common issues, period comparison) that needs custom SQL.",
                "suggested_agent": "data_analyst",
            },
            insights=[
                "This support query requires advanced analytics beyond my core capabilities.",
                "Routing to Data Analyst for custom SQL generation with HITL approval.",
            ],
            recommendations=[],
        )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        query = params.get("query", "")
        mode = params.get("mode", "sentiment_analysis")

        # Check for complex queries first
        if self._is_complex_query(query):
            logger.info("support agent: complex query detected, returning cannot_handle")
            return self._cannot_handle(query)

        if mode == "ticket_trends":
            return await self._run_ticket_trends(params)
        else:
            return await self._run_sentiment_analysis(params)

    async def _run_sentiment_analysis(self, params: dict[str, Any]) -> AgentResult:
        """Analyze support sentiment."""
        window_days = params.get("window_days", 7)
        product_id = params.get("product_id")

        try:
            sentiment = await self.tools.support.get_support_sentiment(
                GetSupportSentimentRequest(window_days=window_days, product_id=product_id)
            )
        except Exception as exc:
            logger.exception("support agent (sentiment_analysis) failed: %s", exc)
            return self.failure(exc)

        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []
        stats = sentiment.sentiment

        insights.append(f"Support sentiment analysis (last {window_days} days):")
        insights.append(f"  Total tickets: {stats.ticket_volume}")
        insights.append(f"  Average sentiment: {stats.avg_sentiment:.2f}")
        insights.append(f"  Negative ratio: {stats.negative_ratio:.0%}")

        # Check for critical sentiment levels
        if stats.avg_sentiment < 0.3:
            insights.append(
                f"🔴 CRITICAL: Average sentiment is {stats.avg_sentiment:.2f} - customer satisfaction is severely impacted."
            )
            recommendations.append(
                AgentRecommendation(
                    action_type="escalate_ticket",
                    payload={
                        "ticket_id": -1,
                        "priority": "critical",
                    },
                    reasoning=f"Sentiment critically low ({stats.avg_sentiment:.2f}). Recommend escalating recent tickets.",
                    requires_approval=True,
                )
            )
        elif stats.avg_sentiment < 0.4:
            insights.append(
                f"🟠 WARNING: Average sentiment {stats.avg_sentiment:.2f} indicates high risk."
            )

        if stats.negative_ratio > 0.5:
            insights.append(f"⚠️ Negative tickets ratio {stats.negative_ratio:.0%} is concerning.")
            if stats.negative_ratio > 0.7:
                recommendations.append(
                    AgentRecommendation(
                        action_type="prioritize_ticket",
                        payload={"ticket_id": -1, "priority": "high"},
                        reasoning=f"Over {stats.negative_ratio:.0%} of tickets are negative. Recommend prioritizing unresolved tickets.",
                        requires_approval=True,
                    )
                )

        findings = sentiment.model_dump()
        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_ticket_trends(self, params: dict[str, Any]) -> AgentResult:
        """Analyze ticket trends."""
        window_days = params.get("window_days", 14)
        group_by = params.get("group_by", "issue_category")
        product_id = params.get("product_id")

        try:
            resp = await self.tools.support.get_ticket_trends(
                GetTicketTrendsRequest(
                    window_days=window_days,
                    group_by=group_by,
                    product_id=product_id,
                )
            )
        except Exception as exc:
            logger.exception("support agent (ticket_trends) failed: %s", exc)
            return self.failure(exc)

        findings = resp.model_dump()
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(
            f"Ticket trends (last {resp.window_days} days, grouped by {resp.group_by}):"
        )
        insights.append(f"  Total volume: {resp.total_volume}")

        for trend in resp.trends[:10]:
            trend_icon = (
                "📈"
                if trend.trend == "increasing"
                else "📉" if trend.trend == "decreasing" else "➡️"
            )
            insights.append(
                f"  {trend_icon} {trend.key}: {trend.volume} tickets ({trend.change_pct:+.1f}%)"
            )
            if trend.avg_sentiment is not None and trend.avg_sentiment < 0.3:
                insights.append(f"      ⚠️ Low sentiment: {trend.avg_sentiment:.2f}")

        if resp.alerts:
            for alert in resp.alerts:
                insights.append(f"🚨 {alert}")

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
