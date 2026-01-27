"""
MCP Action Tools: Write operations for HITL action execution.

These tools perform actual data modifications (inventory updates, campaign changes, etc.)
and should only be invoked after human approval.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from mcp_server.tools.base import BaseTool


# =============================================================================
# INVENTORY ACTION TOOLS
# =============================================================================


class UpdateInventoryPayload(BaseModel):
    product_id: int = Field(..., description="Product ID to update")
    quantity_change: int = Field(..., description="Amount to add (positive) or remove (negative)")
    reason: str | None = Field(None, description="Reason for the adjustment")


class UpdateInventoryTool(BaseTool):
    """Update product inventory stock quantity."""

    name = "update_inventory"

    def request_model(self) -> type[BaseModel]:
        return UpdateInventoryPayload

    async def run(self, session, payload: UpdateInventoryPayload) -> dict[str, Any]:
        # First get current stock
        check_stmt = text("SELECT id, name, stock_qty FROM products WHERE id = :product_id")
        result = await session.execute(check_stmt, {"product_id": payload.product_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Product {payload.product_id} not found"}

        old_qty = row.stock_qty
        new_qty = old_qty + payload.quantity_change

        if new_qty < 0:
            return {
                "success": False,
                "error": f"Cannot reduce stock below 0. Current: {old_qty}, Change: {payload.quantity_change}",
            }

        # Update the stock
        update_stmt = text(
            """
            UPDATE products 
            SET stock_qty = :new_qty 
            WHERE id = :product_id
            RETURNING id, name, stock_qty
        """
        )
        result = await session.execute(
            update_stmt, {"new_qty": new_qty, "product_id": payload.product_id}
        )
        updated = result.one()
        await session.commit()

        return {
            "success": True,
            "product_id": updated.id,
            "product_name": updated.name,
            "old_quantity": old_qty,
            "new_quantity": updated.stock_qty,
            "change": payload.quantity_change,
            "reason": payload.reason,
        }


# =============================================================================
# CAMPAIGN ACTION TOOLS
# =============================================================================


class UpdateCampaignStatusPayload(BaseModel):
    campaign_id: int = Field(..., description="Campaign ID to update")
    status: str = Field(
        ..., pattern="^(active|paused)$", description="New status: 'active' or 'paused'"
    )
    reason: str | None = Field(None, description="Reason for the status change")


class UpdateCampaignStatusTool(BaseTool):
    """Update campaign status (pause/resume)."""

    name = "update_campaign_status"

    def request_model(self) -> type[BaseModel]:
        return UpdateCampaignStatusPayload

    async def run(self, session, payload: UpdateCampaignStatusPayload) -> dict[str, Any]:
        # Get current campaign
        check_stmt = text("SELECT id, name, status FROM campaigns WHERE id = :campaign_id")
        result = await session.execute(check_stmt, {"campaign_id": payload.campaign_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Campaign {payload.campaign_id} not found"}

        old_status = row.status

        # Update status
        update_stmt = text(
            """
            UPDATE campaigns 
            SET status = :new_status 
            WHERE id = :campaign_id
            RETURNING id, name, status
        """
        )
        result = await session.execute(
            update_stmt, {"new_status": payload.status, "campaign_id": payload.campaign_id}
        )
        updated = result.one()
        await session.commit()

        return {
            "success": True,
            "campaign_id": updated.id,
            "campaign_name": updated.name,
            "old_status": old_status,
            "new_status": updated.status,
            "reason": payload.reason,
        }


class UpdateCampaignBudgetPayload(BaseModel):
    campaign_id: int = Field(..., description="Campaign ID to update")
    new_budget: float = Field(..., gt=0, description="New budget amount")
    reason: str | None = Field(None, description="Reason for the budget change")


class UpdateCampaignBudgetTool(BaseTool):
    """Update campaign budget."""

    name = "update_campaign_budget"

    def request_model(self) -> type[BaseModel]:
        return UpdateCampaignBudgetPayload

    async def run(self, session, payload: UpdateCampaignBudgetPayload) -> dict[str, Any]:
        # Get current campaign
        check_stmt = text("SELECT id, name, budget FROM campaigns WHERE id = :campaign_id")
        result = await session.execute(check_stmt, {"campaign_id": payload.campaign_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Campaign {payload.campaign_id} not found"}

        old_budget = float(row.budget)

        # Update budget
        update_stmt = text(
            """
            UPDATE campaigns 
            SET budget = :new_budget 
            WHERE id = :campaign_id
            RETURNING id, name, budget
        """
        )
        result = await session.execute(
            update_stmt, {"new_budget": payload.new_budget, "campaign_id": payload.campaign_id}
        )
        updated = result.one()
        await session.commit()

        return {
            "success": True,
            "campaign_id": updated.id,
            "campaign_name": updated.name,
            "old_budget": old_budget,
            "new_budget": float(updated.budget),
            "reason": payload.reason,
        }


# =============================================================================
# SUPPORT TICKET ACTION TOOLS
# =============================================================================


class EscalateTicketPayload(BaseModel):
    ticket_id: int = Field(..., description="Ticket ID to escalate")
    priority: str = Field(default="high", description="New priority level")
    reason: str | None = Field(None, description="Reason for escalation")


class EscalateTicketTool(BaseTool):
    """Escalate a support ticket to higher priority."""

    name = "escalate_ticket"

    def request_model(self) -> type[BaseModel]:
        return EscalateTicketPayload

    async def run(self, session, payload: EscalateTicketPayload) -> dict[str, Any]:
        # Check ticket exists
        check_stmt = text("SELECT id, issue_category FROM support_tickets WHERE id = :ticket_id")
        result = await session.execute(check_stmt, {"ticket_id": payload.ticket_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Ticket {payload.ticket_id} not found"}

        # Note: In a real system, you'd have a priority column.
        # For now, we'll just return success as the schema doesn't have priority.
        return {
            "success": True,
            "ticket_id": row.id,
            "issue_category": row.issue_category,
            "new_priority": payload.priority,
            "reason": payload.reason,
            "note": "Ticket escalated (priority tracking not yet in schema)",
        }


class CloseTicketPayload(BaseModel):
    ticket_id: int = Field(..., description="Ticket ID to close")
    resolution: str | None = Field(None, description="Resolution summary")


class CloseTicketTool(BaseTool):
    """Close a support ticket."""

    name = "close_ticket"

    def request_model(self) -> type[BaseModel]:
        return CloseTicketPayload

    async def run(self, session, payload: CloseTicketPayload) -> dict[str, Any]:
        # Check ticket exists
        check_stmt = text("SELECT id, issue_category FROM support_tickets WHERE id = :ticket_id")
        result = await session.execute(check_stmt, {"ticket_id": payload.ticket_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Ticket {payload.ticket_id} not found"}

        # Note: In a real system, you'd have a status column for tickets.
        return {
            "success": True,
            "ticket_id": row.id,
            "issue_category": row.issue_category,
            "resolution": payload.resolution,
            "note": "Ticket closed (status tracking not yet in schema)",
        }


class PrioritizeTicketPayload(BaseModel):
    ticket_id: int = Field(..., description="Ticket ID to prioritize")
    priority: str = Field(
        default="medium", description="Priority level: low, medium, high, critical"
    )


class PrioritizeTicketTool(BaseTool):
    """Set priority level for a support ticket."""

    name = "prioritize_ticket"

    def request_model(self) -> type[BaseModel]:
        return PrioritizeTicketPayload

    async def run(self, session, payload: PrioritizeTicketPayload) -> dict[str, Any]:
        check_stmt = text("SELECT id, issue_category FROM support_tickets WHERE id = :ticket_id")
        result = await session.execute(check_stmt, {"ticket_id": payload.ticket_id})
        row = result.one_or_none()

        if row is None:
            return {"success": False, "error": f"Ticket {payload.ticket_id} not found"}

        return {
            "success": True,
            "ticket_id": row.id,
            "issue_category": row.issue_category,
            "priority": payload.priority,
            "note": "Priority set (priority tracking not yet in schema)",
        }
