"""
DataAnalystAgent: Handles complex queries that don't fit specialized agents.

This agent:
1. Receives queries that require custom SQL analysis
2. Generates SQL from natural language (when statement not provided)
3. ALWAYS requires HITL approval before SQL execution
4. Returns findings after approval
"""

from __future__ import annotations

import logging
import re

from prompts import get_prompt, render_prompt
from utils.llm import get_llm
from .base_agent import (
    AgentCapability,
    AgentMetadata,
    AgentRecommendation,
    AgentResult,
    AgentRunContext,
    AgentTask,
    BaseAgent,
)

logger = logging.getLogger("agent.data_analyst")


class DataAnalystAgent(BaseAgent):
    name = "data_analyst"
    description = "Handles complex queries by generating SQL with HITL approval."

    metadata = AgentMetadata(
        name="data_analyst",
        display_name="DATA_ANALYST",
        description=(
            "Performs custom SQL queries for complex analysis. "
            "Used as fallback when specialized agents cannot handle the query. "
            "ALL SQL execution requires human approval (HITL)."
        ),
        capabilities=[
            AgentCapability(
                name="custom_analysis",
                description=(
                    "Generate and execute custom SQL for complex cross-domain analysis. "
                    "Requires HITL approval before execution."
                ),
                parameters={
                    "query": "Natural language description of the analysis needed",
                    "statement": "Optional: Pre-built SQL statement (if provided, skips generation)",
                },
                example_queries=[
                    "Compare yesterday's sales with last week by region",
                    "Which products are driving the most support tickets?",
                    "Show me underperforming campaigns with low conversion rates",
                    "Complex query: revenue by channel and product category",
                ],
            ),
        ],
        keywords=[
            "complex",
            "custom",
            "compare",
            "analyze",
            "breakdown",
            "cross-domain",
            "advanced",
            "report",
            "regional",
            "channel",
        ],
        priority_boost=["complex analysis", "custom report", "compare periods"],
    )

    async def run(self, task: AgentTask, context: AgentRunContext) -> AgentResult:
        """
        Handle custom analysis requests.

        Flow:
        1. If statement provided, use it directly
        2. Otherwise, generate SQL from natural language query
        3. Create HITL recommendation for approval (ALWAYS)
        4. After approval, result will be executed via action_executor
        """
        statement = task.parameters.get("statement")
        query = task.parameters.get("query", task.parameters.get("original_query", ""))

        logger.info(f"DataAnalystAgent received task: query='{query}', statement={bool(statement)}")

        # Step 1: Generate SQL if not provided
        if not statement:
            statement = await self._generate_sql(query, context)
            if statement is None:
                return self.failure(
                    "Could not generate SQL for this query. "
                    "Please provide more details or a specific SQL statement."
                )

        # Step 2: Always create HITL recommendation for custom SQL
        recommendation = AgentRecommendation(
            action_type="execute_custom_sql",
            payload={
                "statement": statement,
                "original_query": query,
            },
            reasoning=(
                f"Custom SQL analysis requested: '{query}'. "
                f"This query requires human approval before execution. "
                f"SQL: {statement[:200]}{'...' if len(statement) > 200 else ''}"
            ),
            requires_approval=True,  # Always require approval for custom SQL
        )

        logger.info("DataAnalystAgent created HITL recommendation for SQL execution")

        return self.success(
            findings={
                "generated_sql": statement,
                "original_query": query,
                "status": "pending_approval",
                "message": "Custom SQL query generated. Awaiting human approval before execution.",
            },
            insights=[
                "Custom SQL analysis requires HITL approval before execution.",
                f"Query: {query}",
            ],
            recommendations=[recommendation],
        )

    async def _generate_sql(self, query: str, context: AgentRunContext) -> str | None:
        """
        Generate SQL from natural language query using the LLM.

        Returns the generated SQL statement or None if generation fails.
        """
        # Load prompt from centralized prompts
        db_schema = get_prompt("db_schema.schema")
        prompt = render_prompt(
            "sql_generation.generate.template",
            db_schema=db_schema,
            query=query,
        )

        try:
            llm = get_llm()
            response = await llm.ainvoke(prompt)
            sql = response.content.strip()

            # Check if LLM couldn't generate
            cannot_generate_marker = get_prompt("sql_generation.cannot_generate_marker")
            if cannot_generate_marker in sql:
                logger.warning(f"LLM could not generate SQL for query: {query}")
                return None

            # Clean up the response - remove markdown code blocks if present
            sql = re.sub(r"^```sql\s*", "", sql)
            sql = re.sub(r"^```\s*", "", sql)
            sql = re.sub(r"\s*```$", "", sql)
            sql = sql.strip()

            # Basic validation - should start with SELECT, WITH, etc.
            if not re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)", sql, re.IGNORECASE):
                logger.warning(
                    f"LLM returned invalid SQL (doesn't start with valid keyword): {sql[:100]}"
                )
                return None

            logger.info(f"LLM generated SQL for query '{query}': {sql[:100]}...")
            return sql

        except Exception as e:
            logger.error(f"LLM SQL generation failed: {e}")
            return None
