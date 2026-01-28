from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.tools import GetSupportSentimentRequest, GetTicketTrendsRequest
from opsbrain_graph.tools.support_tools import (
    GetCommonIssuesRequest,
    GetComplaintTrendsRequest,
)
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


class SupportAgent(BaseAgent):
    name = "support"
    description = "Analyzes support sentiment and issue trends."

    metadata = AgentMetadata(
        name="support",
        display_name="SUPPORT",
        description="Analyzes customer support tickets, sentiment trends, issue patterns, and complaint volumes. Can identify common issues and complaint spikes.",
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
                    "How are support tickets trending?",
                ],
            ),
            AgentCapability(
                name="issue_detection",
                description="Detect spikes in specific issue categories",
                parameters={
                    "window_days": "Time period to analyze",
                    "product_id": "Optional product filter",
                },
                example_queries=[
                    "Are there any issue spikes?",
                    "What are customers complaining about?",
                    "Show me support trends for product X",
                ],
            ),
            AgentCapability(
                name="common_issues",
                description="Find the most common customer issues reported",
                parameters={
                    "window_days": "Time period to analyze (default: 7)",
                    "min_count": "Minimum occurrences to be considered common (default: 2)",
                    "limit": "Max issues to return (default: 10)",
                },
                example_queries=[
                    "Is there a common issue reported by customers?",
                    "What are customers complaining about most?",
                    "Show me top customer issues",
                ],
            ),
            AgentCapability(
                name="complaint_trends",
                description="Compare complaint volume between current and previous periods",
                parameters={
                    "current_days": "Current period in days (default: 1)",
                    "previous_days": "Previous period for comparison (default: 7)",
                    "issue_category": "Optional filter by issue category",
                },
                example_queries=[
                    "Did customer complaints increase yesterday?",
                    "Are complaints trending up or down?",
                    "Compare today's tickets to last week",
                ],
            ),
            AgentCapability(
                name="ticket_escalation",
                description="Recommend escalating or prioritizing tickets based on sentiment patterns",
                parameters={
                    "ticket_id": "Specific ticket ID to escalate",
                    "priority": "New priority level (low, medium, high, critical)",
                },
                example_queries=[
                    "Escalate urgent tickets",
                    "Which tickets need immediate attention?",
                    "Prioritize high-severity issues",
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
            "satisfaction",
            "escalate",
            "priority",
            "common issue",
        ],
        priority_boost=[
            "angry customers",
            "high complaints",
            "negative sentiment",
            "urgent ticket",
        ],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "sentiment_analysis")

        if mode == "common_issues":
            return await self._run_common_issues(params)
        elif mode == "complaint_trends":
            return await self._run_complaint_trends(params)
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

        insights.append(f"Support sentiment analysis for the last {window_days} days:")
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

    async def _run_common_issues(self, params: dict[str, Any]) -> AgentResult:
        """Find common customer issues."""
        window_days = params.get("window_days", 7)
        min_count = params.get("min_count", 2)
        limit = params.get("limit", 10)

        try:
            resp = await self.tools.support.get_common_issues(
                GetCommonIssuesRequest(
                    window_days=window_days,
                    min_count=min_count,
                    limit=limit,
                )
            )
        except Exception as exc:
            logger.exception("support agent (common_issues) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "window_days": resp.window_days,
            "common_issues": [i.model_dump() for i in resp.common_issues],
            "most_common_issue": resp.most_common_issue,
            "total_tickets_analyzed": resp.total_tickets_analyzed,
            "has_critical_issues": resp.has_critical_issues,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"Common issues analysis for the last {window_days} days:")
        insights.append(f"  Total tickets analyzed: {resp.total_tickets_analyzed}")

        if not resp.common_issues:
            insights.append("✅ No common issues detected - complaints are diverse.")
        else:
            if resp.most_common_issue:
                insights.append(f"📌 Most common issue: {resp.most_common_issue}")

            for issue in resp.common_issues:
                sentiment_icon = (
                    "🔴"
                    if issue.avg_sentiment < 0.3
                    else "🟠" if issue.avg_sentiment < 0.5 else "🟢"
                )
                insights.append(
                    f"  {sentiment_icon} {issue.issue_category}: {issue.ticket_count} tickets "
                    f"(sentiment: {issue.avg_sentiment:.2f}, negative: {issue.negative_ratio:.0%})"
                )
                if issue.sample_descriptions:
                    for desc in issue.sample_descriptions[:2]:
                        insights.append(
                            f'      └ "{desc[:100]}..."' if len(desc) > 100 else f'      └ "{desc}"'
                        )

            if resp.has_critical_issues:
                insights.append("⚠️ Critical issues detected requiring attention!")

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_complaint_trends(self, params: dict[str, Any]) -> AgentResult:
        """Compare complaint volume between periods."""
        current_days = params.get("current_days", 1)
        previous_days = params.get("previous_days", 7)
        issue_category = params.get("issue_category")

        try:
            resp = await self.tools.support.get_complaint_trends(
                GetComplaintTrendsRequest(
                    current_days=current_days,
                    previous_days=previous_days,
                    issue_category=issue_category,
                )
            )
        except Exception as exc:
            logger.exception("support agent (complaint_trends) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "current_days": resp.current_days,
            "previous_days": resp.previous_days,
            "current_total": resp.current_total,
            "previous_avg": resp.previous_avg,
            "overall_change_pct": resp.overall_change_pct,
            "complaint_increased": resp.complaint_increased,
            "category_trends": [c.model_dump() for c in resp.category_trends],
            "most_increased_category": resp.most_increased_category,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(
            f"Complaint trends (last {current_days} day(s) vs {previous_days} days average):"
        )
        insights.append(f"  Current complaints: {resp.current_total}")
        insights.append(f"  Previous daily average: {resp.previous_avg:.1f}")

        if resp.complaint_increased:
            insights.append(f"📈 Complaints INCREASED by {resp.overall_change_pct:+.1f}%")
            if resp.most_increased_category:
                insights.append(f"  Most increased category: {resp.most_increased_category}")

            # Add recommendation if significant increase
            if resp.overall_change_pct > 50:
                recommendations.append(
                    AgentRecommendation(
                        action_type="investigate_complaints",
                        payload={
                            "change_pct": resp.overall_change_pct,
                            "top_category": resp.most_increased_category,
                        },
                        reasoning=f"Complaints increased by {resp.overall_change_pct:.1f}%, investigate root cause",
                        requires_approval=False,
                    )
                )
        else:
            insights.append(f"📉 Complaints decreased by {abs(resp.overall_change_pct):.1f}%")

        if resp.current_avg_sentiment is not None and resp.previous_avg_sentiment is not None:
            insights.append(
                f"  Sentiment: {resp.current_avg_sentiment:.2f} (was {resp.previous_avg_sentiment:.2f})"
            )

        # Show category breakdowns
        if resp.category_trends:
            insights.append("Category breakdown:")
            for cat in resp.category_trends[:5]:
                trend_icon = (
                    "📈"
                    if cat.trend == "increasing"
                    else "📉" if cat.trend == "decreasing" else "➡️"
                )
                insights.append(
                    f"  {trend_icon} {cat.issue_category}: {cat.current_count} ({cat.change_pct:+.1f}%)"
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
