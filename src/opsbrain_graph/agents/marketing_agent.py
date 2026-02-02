"""Marketing Agent - Evaluates campaign spend and ROAS."""

from __future__ import annotations

import logging
from typing import Any

from opsbrain_graph.tools import GetCampaignSpendRequest
from opsbrain_graph.tools.marketing_tools import CalculateROASRequest
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
    """Marketing Agent with 4 core capabilities."""

    name = "marketing"
    description = "Evaluates campaign spend, ROAS, and manages campaign status."

    metadata = AgentMetadata(
        name="marketing",
        display_name="MARKETING",
        description="Evaluates marketing campaign spend, calculates ROAS, and manages campaign status (pause/resume).",
        capabilities=[
            AgentCapability(
                name="campaign_spend",
                description="Get campaign spend, clicks, and conversion metrics",
                parameters={
                    "campaign_ids": "Optional list of specific campaign IDs",
                    "status": "Filter by status: active, paused (optional)",
                },
                example_queries=[
                    "How much have we spent on campaigns?",
                    "Show me campaign metrics",
                    "What's our ad spend?",
                ],
            ),
            AgentCapability(
                name="calculate_roas",
                description="Calculate Return on Ad Spend for campaigns",
                parameters={
                    "campaign_id": "Optional specific campaign ID",
                    "window_days": "Analysis window (default: 7)",
                },
                example_queries=[
                    "What's our ROAS?",
                    "Calculate return on ad spend",
                    "Campaign efficiency metrics",
                ],
            ),
            AgentCapability(
                name="pause_campaign",
                description="Pause a marketing campaign by name or ID",
                parameters={
                    "campaign_name": "Name of the campaign to pause",
                    "campaign_id": "ID of the campaign to pause (optional if name provided)",
                },
                example_queries=[
                    "Pause the campaign Spring Hydration Push",
                    "Can you pause campaign 5?",
                    "Stop the Summer Sale campaign",
                    "Pause this campaign",
                ],
            ),
            AgentCapability(
                name="resume_campaign",
                description="Resume a paused marketing campaign by name or ID",
                parameters={
                    "campaign_name": "Name of the campaign to resume",
                    "campaign_id": "ID of the campaign to resume (optional if name provided)",
                },
                example_queries=[
                    "Resume the campaign Spring Hydration Push",
                    "Can you resume campaign 5?",
                    "Reactivate the Summer Sale campaign",
                ],
            ),
        ],
        keywords=[
            "campaign",
            "ad",
            "marketing",
            "roas",
            "spend",
            "advertising",
            "budget",
            "pause",
            "resume",
            "stop",
        ],
        priority_boost=["wasted spend", "low roas", "pause campaign", "stop campaign"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        """Execute the marketing agent task based on mode."""
        params = task.parameters
        mode = params.get("mode", "campaign_spend")

        if mode == "calculate_roas":
            return await self._run_calculate_roas(params)
        elif mode == "pause_campaign":
            return await self._run_pause_campaign(params)
        elif mode == "resume_campaign":
            return await self._run_resume_campaign(params)
        else:
            return await self._run_campaign_spend(params)

    async def _run_campaign_spend(self, params: dict[str, Any]) -> AgentResult:
        """Get campaign spend metrics."""
        campaign_ids = params.get("campaign_ids")
        status = params.get("status")

        try:
            spend = await self.tools.marketing.get_campaign_spend(
                GetCampaignSpendRequest(campaign_ids=campaign_ids, status=status)
            )
        except Exception as exc:
            logger.exception("marketing agent (campaign_spend) failed: %s", exc)
            return self.failure(exc)

        findings = spend.model_dump()
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append("Campaign spend metrics:")
        insights.append(f"  Total spend: ${spend.summary.get('total_spend', 0):,.2f}")
        insights.append(f"  Total campaigns: {spend.campaign_count}")

        for campaign in spend.campaigns:
            utilization = campaign.budget_utilization_pct
            if utilization > 90:
                insights.append(
                    f"⚠️ Campaign {campaign.name} is at {utilization:.0f}% budget utilization"
                )
            if campaign.status == "active" and campaign.spend > 0 and campaign.conversions == 0:
                insights.append(
                    f"⚠️ Campaign {campaign.name} spending ${campaign.spend:.2f} with 0 conversions."
                )
                recommendations.append(
                    AgentRecommendation(
                        action_type="pause_campaign",
                        payload={"campaign_id": campaign.campaign_id},
                        reasoning="Spend detected with zero conversions.",
                        requires_approval=True,
                    )
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_calculate_roas(self, params: dict[str, Any]) -> AgentResult:
        """Calculate ROAS for campaigns."""
        campaign_id = params.get("campaign_id")
        window_days = params.get("window_days", 7)

        try:
            resp = await self.tools.marketing.calculate_roas(
                CalculateROASRequest(campaign_id=campaign_id, window_days=window_days)
            )
        except Exception as exc:
            logger.exception("marketing agent (calculate_roas) failed: %s", exc)
            return self.failure(exc)

        findings = resp.model_dump()
        insights: list[str] = []
        recommendations: list[AgentRecommendation] = []

        insights.append(f"ROAS analysis (last {resp.window_days} days):")
        insights.append(f"  Overall ROAS: {resp.overall_roas:.2f}x")
        insights.append(f"  Total spend: ${resp.total_spend:,.2f}")
        insights.append(f"  Estimated revenue: ${resp.total_estimated_revenue:,.2f}")

        for campaign in resp.campaigns:
            perf_icon = (
                "🟢"
                if campaign.performance == "excellent"
                else (
                    "🟡"
                    if campaign.performance == "good"
                    else "🟠" if campaign.performance == "break_even" else "🔴"
                )
            )
            insights.append(
                f"  {perf_icon} {campaign.campaign_name}: ROAS {campaign.roas:.2f}x "
                f"(${campaign.spend:,.2f} spend, {campaign.conversions} conversions)"
            )

            if campaign.performance == "poor" and campaign.status == "active":
                recommendations.append(
                    AgentRecommendation(
                        action_type="pause_campaign",
                        payload={"campaign_id": campaign.campaign_id},
                        reasoning=f"Poor ROAS of {campaign.roas:.2f}x with ${campaign.spend:.2f} spend",
                        requires_approval=True,
                    )
                )

        return self.success(findings=findings, insights=insights, recommendations=recommendations)

    async def _run_pause_campaign(self, params: dict[str, Any]) -> AgentResult:
        """Pause a campaign by name or ID."""
        return await self._run_campaign_status_change(params, target_status="paused")

    async def _run_resume_campaign(self, params: dict[str, Any]) -> AgentResult:
        """Resume a paused campaign by name or ID."""
        return await self._run_campaign_status_change(params, target_status="active")

    async def _run_campaign_status_change(
        self, params: dict[str, Any], target_status: str
    ) -> AgentResult:
        """Handle campaign status change (pause/resume) by name or ID."""
        campaign_name = params.get("campaign_name")
        campaign_id = params.get("campaign_id")
        action_word = "Pause" if target_status == "paused" else "Resume"
        action_type = "pause_campaign" if target_status == "paused" else "resume_campaign"

        # Get all campaigns to find by name or ID
        try:
            spend_resp = await self.tools.marketing.get_campaign_spend(
                GetCampaignSpendRequest(campaign_ids=None, status=None)
            )
        except Exception as exc:
            logger.exception("marketing agent (campaign status change) failed: %s", exc)
            return self.failure(exc)

        # Find the campaign
        campaign = None
        if campaign_id:
            campaign = next((c for c in spend_resp.campaigns if c.campaign_id == campaign_id), None)
        elif campaign_name:
            # Case-insensitive partial match
            campaign = next(
                (c for c in spend_resp.campaigns if campaign_name.lower() in c.name.lower()),
                None,
            )

        if campaign is None:
            search_term = f"ID {campaign_id}" if campaign_id else f"'{campaign_name}'"
            return self.success(
                findings={"error": f"Campaign {search_term} not found"},
                insights=[f"❌ Could not find campaign matching {search_term}"],
                recommendations=[],
            )

        # Check if already in target status
        if campaign.status == target_status:
            status_word = "paused" if target_status == "paused" else "active"
            return self.success(
                findings={
                    "campaign_id": campaign.campaign_id,
                    "campaign_name": campaign.name,
                    "status": campaign.status,
                },
                insights=[
                    f"ℹ️ Campaign **{campaign.name}** (ID: {campaign.campaign_id}) is already {status_word}."
                ],
                recommendations=[],
            )

        # Create status change recommendation
        findings = {
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.name,
            "current_status": campaign.status,
            "target_status": target_status,
        }
        insights = [
            f"📢 {action_word} request for campaign **{campaign.name}** (ID: {campaign.campaign_id})",
            f"   Current status: {campaign.status}",
            f"   Spend: ${campaign.spend:,.2f} | Clicks: {campaign.clicks} | Conversions: {campaign.conversions}",
        ]
        recommendations = [
            AgentRecommendation(
                action_type=action_type,
                payload={"campaign_id": campaign.campaign_id},
                reasoning=f"User requested to {action_word.lower()} campaign {campaign.name}",
                requires_approval=True,
            )
        ]

        return self.success(findings=findings, insights=insights, recommendations=recommendations)
