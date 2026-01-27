from __future__ import annotations

from opsbrain_graph.tools import GetCampaignSpendRequest
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)


class MarketingAgent(BaseAgent):
    name = "marketing"
    description = "Evaluates campaign performance and spend efficiency."

    metadata = AgentMetadata(
        name="marketing",
        display_name="MARKETING",
        description="Evaluates marketing campaign performance, ad spend efficiency, and ROI. Can identify underperforming campaigns.",
        capabilities=[
            AgentCapability(
                name="campaign_analysis",
                description="Analyze campaign spend, clicks, conversions, and ROAS",
                parameters={
                    "campaign_ids": "Optional list of specific campaign IDs (default: all)",
                    "window_days": "Time period to analyze (default: 7)",
                },
                example_queries=[
                    "How are our marketing campaigns performing?",
                    "What's our ad spend efficiency?",
                    "Which campaigns have low ROAS?",
                ],
            ),
            AgentCapability(
                name="budget_check",
                description="Check campaign budgets and spend rates",
                parameters={
                    "campaign_ids": "Optional list of campaign IDs",
                },
                example_queries=[
                    "Are any campaigns overspending?",
                    "Show me campaign budgets vs actual spend",
                ],
            ),
        ],
        keywords=[
            "campaign",
            "ad",
            "marketing",
            "roas",
            "spend",
            "promotion",
            "advertising",
            "budget",
        ],
        priority_boost=["wasted spend", "zero conversions", "overspending"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext):
        campaign_ids = task.parameters.get("campaign_ids")
        window_days = task.parameters.get("window_days", 7)

        try:
            spend = await self.tools.marketing.get_campaign_spend(
                GetCampaignSpendRequest(campaign_ids=campaign_ids, window_days=window_days)
            )
        except Exception as exc:
            return self.failure(exc)

        findings = spend.model_dump()
        insights = []

        for campaign in spend.campaigns:
            if campaign.status == "active" and campaign.spend > 0 and campaign.conversions == 0:
                insights.append(
                    f"Campaign {campaign.name} (ID: {campaign.id}) spending ${campaign.spend:.2f} with 0 conversions."
                )
                recommendations = [
                    AgentRecommendation(
                        action_type="pause_campaign",
                        payload={"campaign_id": campaign.id},
                        reasoning="Spend detected with zero conversions.",
                        requires_approval=True,
                    )
                ]
                return self.success(
                    findings=findings, insights=insights, recommendations=recommendations
                )

        return self.success(findings=findings, insights=insights)
