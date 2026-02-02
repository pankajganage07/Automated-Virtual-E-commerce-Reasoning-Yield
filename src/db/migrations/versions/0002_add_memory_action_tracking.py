"""add action tracking columns to agent_memory

Revision ID: 0002
Revises: 0001
Create Date: 2024-06-02 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add structured action tracking columns to agent_memory table.

    These columns enable the system to answer "What did we do last time?" queries
    by storing:
    - actions_proposed: All actions recommended by agents
    - actions_approved: Action types that were approved via HITL
    - actions_rejected: Action types that were rejected via HITL
    - confidence_score: The confidence level of the analysis
    """
    op.add_column(
        "agent_memory",
        sa.Column("actions_proposed", JSONB, nullable=True),
    )
    op.add_column(
        "agent_memory",
        sa.Column("actions_approved", JSONB, nullable=True),
    )
    op.add_column(
        "agent_memory",
        sa.Column("actions_rejected", JSONB, nullable=True),
    )
    op.add_column(
        "agent_memory",
        sa.Column("confidence_score", sa.Float, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_memory", "confidence_score")
    op.drop_column("agent_memory", "actions_rejected")
    op.drop_column("agent_memory", "actions_approved")
    op.drop_column("agent_memory", "actions_proposed")
