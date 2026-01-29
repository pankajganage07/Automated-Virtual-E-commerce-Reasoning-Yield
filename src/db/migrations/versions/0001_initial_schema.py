"""initial schema

Revision ID: 0001
Revises:
Create Date: 2024-06-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')

    op.create_table(
        "products",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("price", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("stock_qty", sa.Integer, nullable=False, server_default="0"),
        sa.Column("low_stock_threshold", sa.Integer, nullable=False, server_default="10"),
    )

    op.create_table(
        "inventory",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "product_id",
            sa.Integer,
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("warehouse_code", sa.String(length=50), nullable=False),
        sa.Column("on_hand", sa.Integer, nullable=False, server_default="0"),
        sa.Column("reserved", sa.Integer, nullable=False, server_default="0"),
        sa.Column("reorder_point", sa.Integer, nullable=False, server_default="0"),
        sa.Column("incoming_qty", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_restocked", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("product_id", "warehouse_code", name="uq_inventory_product_warehouse"),
        sa.Index("idx_inventory_product_id", "product_id"),
    )

    op.create_table(
        "campaigns",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("budget", sa.Numeric(12, 2), nullable=False),
        sa.Column("spend", sa.Numeric(12, 2), nullable=False, server_default="0"),
        sa.Column("clicks", sa.Integer, nullable=False, server_default="0"),
        sa.Column("conversions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "product_id",
            sa.Integer,
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("qty", sa.Integer, nullable=False),
        sa.Column("revenue", sa.Numeric(12, 2), nullable=False),
        sa.Column("region", sa.String(length=100), nullable=False),
        sa.Column("channel", sa.String(length=100), nullable=False),
    )

    op.create_table(
        "support_tickets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "product_id",
            sa.Integer,
            sa.ForeignKey("products.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("sentiment", sa.Float, nullable=False),
        sa.Column("issue_category", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "agent_memory",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("incident_summary", sa.Text, nullable=False),
        sa.Column("root_cause", sa.Text),
        sa.Column("action_taken", sa.Text),
        sa.Column("outcome", sa.Text),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("agent_name", sa.String(100), nullable=False),
        sa.Column("action_type", sa.String(100), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("reasoning", sa.Text, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_pending_actions_status", "pending_actions", ["status"])


def downgrade() -> None:
    op.drop_index("idx_pending_actions_status", table_name="pending_actions")
    op.drop_table("pending_actions")
    op.drop_table("agent_memory")
    op.drop_table("support_tickets")
    op.drop_table("orders")
    op.drop_table("campaigns")
    op.drop_index("idx_inventory_product_id", table_name="inventory")
    op.drop_constraint("uq_inventory_product_warehouse", "inventory", type_="unique")
    op.drop_table("inventory")
    op.drop_table("products")
    op.execute('DROP EXTENSION IF EXISTS "vector";')
