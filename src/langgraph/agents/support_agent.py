from __future__ import annotations

from langgraph.tools import AnalyzeSentimentRequest
from .base_agent import AgentRunContext, AgentTask, BaseAgent


class SupportAgent(BaseAgent):
    name = "support"
    description = "Analyzes support sentiment and issue trends."

    async def run(self, task: AgentTask, context: AgentRunContext):
        window_days = task.parameters.get("window_days", 7)
        product_id = task.parameters.get("product_id")

        try:
            sentiment = await self.tools.support.analyze_sentiment(
                AnalyzeSentimentRequest(window_days=window_days, product_id=product_id)
            )
        except Exception as exc:
            return self.failure(exc)

        insights = []
        stats = sentiment.sentiment
        if stats.avg_sentiment < 0.4:
            insights.append(f"Average sentiment {stats.avg_sentiment:.2f}, high risk.")
        if stats.negative_ratio > 0.5:
            insights.append(
                f"Negative tickets ratio {stats.negative_ratio:.0%} in last {window_days} days."
            )

        findings = sentiment.model_dump()
        return self.success(findings=findings, insights=insights)
