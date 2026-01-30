"""
Marketing Agent - Slimmed architecture (2 core capabilities).

Capabilities:
1. campaign_spend - Get campaign spend and conversion metrics
2. calculate_roas - Calculate Return on Ad Spend

Complex queries (underperforming, comparison) route to DataAnalystAgent.
"""

from __future__ import annotations

import logging
import re
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


# Query patterns that this agent CANNOT handle (require DataAnalystAgent)
COMPLEX_QUERY_PATTERNS = [
    r"underperform",
    r"poor.*(campaign|roas)",
    r"zero.*conversion",
    r"compare.*(campaign|performance|period)",
    r"yesterday.*vs.*week",
    r"campaign.*trend",
    r"performance.*drop",
    r"performance.*improve",
    r"campaign.*comparison",
    r"historical.*campaign",
    r"best.*campaign",
    r"worst.*campaign",
    r"rank.*campaign",
    r"top.*performer",
    r"bottom.*performer",
]


class MarketingAgent(BaseAgent):
    """
    Marketing Agent with 2 core capabilities.

    Complex queries trigger cannot_handle for routing to DataAnalystAgent.
    """

    name = "marketing"
    description = "Evaluates campaign spend and ROAS."

    metadata = AgentMetadata(
        name="marketing",
        display_name="MARKETING",
        description="Evaluates marketing campaign spend and calculates ROAS. For complex analytics (underperforming, comparisons), use data analyst.",
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
        ],
        keywords=[
            "campaign",
            "ad",
            "marketing",
            "roas",
            "spend",
            "advertising",
            "budget",
        ],
        priority_boost=["wasted spend", "low roas"],
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
                "reason": "This query requires complex marketing analysis (comparison, ranking, underperforming) that needs custom SQL.",
                "suggested_agent": "data_analyst",
            },
            insights=[
                "This marketing query requires advanced analytics beyond my core capabilities.",
                "Routing to Data Analyst for custom SQL generation with HITL approval.",
            ],
            recommendations=[],
        )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        params = task.parameters
        query = params.get("query", "")
        mode = params.get("mode", "campaign_spend")

        # Check for complex queries first
        if self._is_complex_query(query):
            logger.info("marketing agent: complex query detected, returning cannot_handle")
            return self._cannot_handle(query)

        if mode == "calculate_roas":
            return await self._run_calculate_roas(params)
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
