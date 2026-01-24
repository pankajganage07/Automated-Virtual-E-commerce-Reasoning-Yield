from __future__ import annotations

from opsbrain_graph.tools import GetAdSpendRequest
from .base_agent import AgentRecommendation, AgentRunContext, AgentTask, BaseAgent


class MarketingAgent(BaseAgent):
    name = "marketing"
    description = "Evaluates campaign performance and spend efficiency."

    async def run(self, task: AgentTask, context: AgentRunContext):
        campaign_ids = task.parameters.get("campaign_ids")
        window_days = task.parameters.get("window_days", 7)

        try:
            spend = await self.tools.marketing.get_ad_spend(
                GetAdSpendRequest(campaign_ids=campaign_ids, window_days=window_days)
            )
        except Exception as exc:
            return self.failure(exc)

        findings = spend.model_dump()
        insights = []

        for campaign in spend.campaigns:
            if campaign.status == "active" and campaign.spend > 0 and campaign.conversions == 0:
                insights.append(
                    f"Campaign {campaign.campaign_id} spending {campaign.spend} with 0 conversions."
                )
                recommendations = [
                    AgentRecommendation(
                        action_type="pause_campaign",
                        payload={"campaign_id": campaign.campaign_id},
                        reasoning="Spend detected with zero conversions.",
                        requires_approval=True,
                    )
                ]
                return self.success(
                    findings=findings, insights=insights, recommendations=recommendations
                )

        return self.success(findings=findings, insights=insights)
