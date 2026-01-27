from __future__ import annotations

from opsbrain_graph.tools import GetSupportSentimentRequest, GetTicketTrendsRequest
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)


class SupportAgent(BaseAgent):
    name = "support"
    description = "Analyzes support sentiment and issue trends."

    metadata = AgentMetadata(
        name="support",
        display_name="SUPPORT",
        description="Analyzes customer support tickets, sentiment trends, and issue patterns. Identifies products with high complaint volumes and can recommend ticket escalation.",
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
        ],
        priority_boost=[
            "angry customers",
            "high complaints",
            "negative sentiment",
            "urgent ticket",
        ],
    )

    async def run(self, task: AgentTask, context: AgentRunContext):
        window_days = task.parameters.get("window_days", 7)
        product_id = task.parameters.get("product_id")

        try:
            sentiment = await self.tools.support.get_support_sentiment(
                GetSupportSentimentRequest(window_days=window_days, product_id=product_id)
            )
        except Exception as exc:
            return self.failure(exc)

        insights = []
        recommendations = []
        stats = sentiment.sentiment

        # Check for critical sentiment levels
        if stats.avg_sentiment < 0.3:
            insights.append(
                f"CRITICAL: Average sentiment is {stats.avg_sentiment:.2f} - customer satisfaction is severely impacted."
            )
            # Could recommend escalating the most recent tickets when sentiment is critical
            recommendations.append(
                AgentRecommendation(
                    action_type="escalate_ticket",
                    payload={
                        "ticket_id": -1,
                        "priority": "critical",
                    },  # -1 indicates batch escalation needed
                    reasoning=f"Sentiment critically low ({stats.avg_sentiment:.2f}). Recommend escalating recent tickets.",
                    requires_approval=True,
                )
            )
        elif stats.avg_sentiment < 0.4:
            insights.append(f"Average sentiment {stats.avg_sentiment:.2f}, high risk.")

        if stats.negative_ratio > 0.5:
            insights.append(
                f"Negative tickets ratio {stats.negative_ratio:.0%} in last {window_days} days."
            )
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
