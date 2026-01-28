from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.tools import GetCampaignSpendRequest
from opsbrain_graph.tools.marketing_tools import (
    GetUnderperformingCampaignsRequest,
    CompareCampaignPerformanceRequest,
    CalculateROASRequest,
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

logger = logging.getLogger("agent.marketing")


class MarketingAgent(BaseAgent):
    name = "marketing"
    description = "Evaluates campaign performance and spend efficiency."

    metadata = AgentMetadata(
        name="marketing",
        display_name="MARKETING",
        description="Evaluates marketing campaign performance, ad spend efficiency, ROAS, and identifies underperforming campaigns. Can compare performance across periods.",
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
            AgentCapability(
                name="underperforming",
                description="Find paused, zero-conversion, or low-ROAS campaigns",
                parameters={
                    "min_spend": "Minimum spend to consider (default: 0)",
                    "include_paused": "Include paused campaigns (default: true)",
                },
                example_queries=[
                    "Were any campaigns paused or underperforming?",
                    "Which campaigns are wasting money?",
                    "Find campaigns with zero conversions",
                ],
            ),
            AgentCapability(
                name="compare_performance",
                description="Compare campaign performance between time periods",
                parameters={
                    "current_days": "Current period in days (default: 1)",
                    "previous_days": "Previous period for comparison (default: 7)",
                },
                example_queries=[
                    "Did campaign performance drop compared to last week?",
                    "How do campaigns compare today vs last week?",
                    "Campaign performance trends",
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
            "underperforming",
            "paused",
        ],
        priority_boost=["wasted spend", "zero conversions", "overspending"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        mode = params.get("mode", "campaign_analysis")

        if mode == "underperforming":
            return await self._run_underperforming(params)
        elif mode == "compare_performance":
            return await self._run_compare_performance(params)
        else:
            return await self._run_campaign_analysis(params)

    async def _run_campaign_analysis(self, params: dict[str, Any]) -> AgentResult:
        """Analyze campaign performance."""
        campaign_ids = params.get("campaign_ids")
        window_days = params.get("window_days", 7)

        try:
            spend = await self.tools.marketing.get_campaign_spend(
                GetCampaignSpendRequest(campaign_ids=campaign_ids, window_days=window_days)
            )
        except Exception as exc:
            logger.exception("marketing agent (campaign_analysis) failed: %s", exc)
            return self.failure(exc)

        findings = spend.model_dump()
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"Campaign analysis for the last {window_days} days:")
        insights.append(f"  Total spend: ${spend.summary.get('total_spend', 0):,.2f}")
        insights.append(f"  Total campaigns: {len(spend.campaigns)}")

        for campaign in spend.campaigns:
            if campaign.status == "active" and campaign.spend > 0 and campaign.conversions == 0:
                insights.append(
                    f"⚠️ Campaign {campaign.name} (ID: {campaign.id}) spending ${campaign.spend:.2f} with 0 conversions."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="pause_campaign",
                        payload={"campaign_id": campaign.id},
                        reasoning="Spend detected with zero conversions.",
                        requires_approval=True,
                    )
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_underperforming(self, params: dict[str, Any]) -> AgentResult:
        """Find underperforming or paused campaigns."""
        min_spend = params.get("min_spend", 0)
        include_paused = params.get("include_paused", True)

        try:
            resp = await self.tools.marketing.get_underperforming_campaigns(
                GetUnderperformingCampaignsRequest(
                    min_spend=min_spend,
                    include_paused=include_paused,
                )
            )
        except Exception as exc:
            logger.exception("marketing agent (underperforming) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "underperforming_campaigns": [c.model_dump() for c in resp.underperforming_campaigns],
            "total_count": resp.total_count,
            "paused_count": resp.paused_count,
            "zero_conversion_count": resp.zero_conversion_count,
            "poor_roas_count": resp.poor_roas_count,
            "has_issues": resp.has_issues,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        if not resp.has_issues:
            insights.append("✅ All campaigns are performing well. No underperformers detected.")
        else:
            insights.append(f"Found {resp.total_count} underperforming campaigns:")

            if resp.paused_count > 0:
                insights.append(f"  ⏸️ {resp.paused_count} campaigns are PAUSED")
            if resp.zero_conversion_count > 0:
                insights.append(
                    f"  🚫 {resp.zero_conversion_count} campaigns have ZERO conversions"
                )
            if resp.poor_roas_count > 0:
                insights.append(f"  📉 {resp.poor_roas_count} campaigns have POOR ROAS")

            for campaign in resp.underperforming_campaigns:
                issue_icon = (
                    "⏸️"
                    if campaign.status == "paused"
                    else "🚫" if campaign.conversions == 0 else "📉"
                )
                insights.append(
                    f"  {issue_icon} {campaign.name}: {campaign.issue} "
                    f"(Spend: ${campaign.spend:,.2f}, ROAS: {campaign.roas:.2f})"
                )

                if campaign.status == "active" and campaign.conversions == 0:
                    recommendations.append(
                        AgentRecommendation(
                            action_type="pause_campaign",
                            payload={"campaign_id": campaign.campaign_id},
                            reasoning=f"Campaign has spent ${campaign.spend:.2f} with zero conversions",
                            requires_approval=True,
                        )
                    )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_compare_performance(self, params: dict[str, Any]) -> AgentResult:
        """Compare campaign performance between periods."""
        current_days = params.get("current_days", 1)
        previous_days = params.get("previous_days", 7)
        campaign_ids = params.get("campaign_ids")

        try:
            resp = await self.tools.marketing.compare_campaign_performance(
                CompareCampaignPerformanceRequest(
                    current_days=current_days,
                    previous_days=previous_days,
                    campaign_ids=campaign_ids,
                )
            )
        except Exception as exc:
            logger.exception("marketing agent (compare_performance) failed: %s", exc)
            return self.failure(exc)

        findings: dict[str, Any] = {
            "current_days": resp.current_days,
            "previous_days": resp.previous_days,
            "campaigns": [c.model_dump() for c in resp.campaigns],
            "total_current_spend": resp.total_current_spend,
            "total_previous_spend": resp.total_previous_spend,
            "overall_spend_change_pct": resp.overall_spend_change_pct,
            "total_current_conversions": resp.total_current_conversions,
            "total_previous_conversions": resp.total_previous_conversions,
            "overall_conversion_change_pct": resp.overall_conversion_change_pct,
            "declining_campaigns_count": resp.declining_campaigns_count,
        }
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(
            f"Campaign performance comparison (last {current_days} day(s) vs {previous_days} days):"
        )
        insights.append(
            f"  Total spend: ${resp.total_current_spend:,.2f} ({resp.overall_spend_change_pct:+.1f}%)"
        )
        insights.append(
            f"  Total conversions: {resp.total_current_conversions} ({resp.overall_conversion_change_pct:+.1f}%)"
        )

        if resp.declining_campaigns_count > 0:
            insights.append(f"⚠️ {resp.declining_campaigns_count} campaigns are declining:")
            for c in resp.campaigns:
                if c.trend == "declining":
                    insights.append(
                        f"    📉 {c.name}: ROAS {c.previous_roas:.2f} → {c.current_roas:.2f} "
                        f"({c.roas_change_pct:+.1f}%)"
                    )
        else:
            insights.append("✅ No campaigns are significantly declining.")

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
