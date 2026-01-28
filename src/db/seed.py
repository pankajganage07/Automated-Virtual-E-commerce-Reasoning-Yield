"""
Database seeding script that loads mock data from CSV files.

Usage:
    python -m db.seed          # Seed all tables
    python -m db.seed --clear  # Clear all tables without seeding
"""

import asyncio
import csv
import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Campaign, Order, Product, SupportTicket, AgentMemory, PendingAction
from db.session import async_session_factory

# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_csv(filename: str) -> list[dict[str, Any]]:
    """Load data from a CSV file in the fixtures directory."""
    filepath = FIXTURES_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found, skipping...")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


async def seed(clear_only: bool = False) -> None:
    """Main seeding function."""
    async with async_session_factory() as session:
        await _clear_tables(session)

        if clear_only:
            print("All tables cleared.")
            return

        await _seed_products(session)
        await session.commit()  # Commit products first for FK constraints

        await _seed_campaigns(session)
        await _seed_orders(session)
        await _seed_support_tickets(session)
        await session.commit()

        # Reset sequences to max ID + 1
        await _reset_sequences(session)
        await session.commit()

        print("✅ Database seeded successfully with mock data from CSV files.")
        await _print_summary(session)


async def _clear_tables(session: AsyncSession) -> None:
    """Clear all tables in correct order (respecting FK constraints)."""
    tables = [PendingAction, AgentMemory, SupportTicket, Order, Campaign, Product]
    for model in tables:
        await session.execute(delete(model))
    await session.commit()
    print("🗑️  Cleared all existing data.")


async def _seed_products(session: AsyncSession) -> None:
    """Seed products from CSV."""
    rows = load_csv("products.csv")
    if not rows:
        return

    products = [
        Product(
            id=int(row["id"]),
            name=row["name"],
            category=row["category"],
            price=Decimal(row["price"]),
            stock_qty=int(row["stock_qty"]),
            low_stock_threshold=int(row["low_stock_threshold"]),
        )
        for row in rows
    ]
    session.add_all(products)
    print(f"📦 Loaded {len(products)} products.")


async def _seed_campaigns(session: AsyncSession) -> None:
    """Seed campaigns from CSV."""
    rows = load_csv("campaigns.csv")
    if not rows:
        return

    campaigns = [
        Campaign(
            id=int(row["id"]),
            name=row["name"],
            budget=Decimal(row["budget"]),
            spend=Decimal(row["spend"]),
            clicks=int(row["clicks"]),
            conversions=int(row["conversions"]),
            status=row["status"],
        )
        for row in rows
    ]
    session.add_all(campaigns)
    print(f"📣 Loaded {len(campaigns)} campaigns.")


async def _seed_orders(session: AsyncSession) -> None:
    """Seed orders from CSV with relative timestamps."""
    rows = load_csv("orders.csv")
    if not rows:
        return

    now = dt.datetime.now(dt.timezone.utc)

    orders = [
        Order(
            id=int(row["id"]),
            product_id=int(row["product_id"]),
            qty=int(row["qty"]),
            revenue=Decimal(row["revenue"]),
            region=row["region"],
            channel=row["channel"],
            timestamp=now - dt.timedelta(days=int(row["days_ago"])),
        )
        for row in rows
    ]
    session.add_all(orders)
    print(f"🛒 Loaded {len(orders)} orders.")


async def _seed_support_tickets(session: AsyncSession) -> None:
    """Seed support tickets from CSV with relative timestamps."""
    rows = load_csv("support_tickets.csv")
    if not rows:
        return

    now = dt.datetime.now(dt.timezone.utc)

    tickets = [
        SupportTicket(
            id=int(row["id"]),
            product_id=int(row["product_id"]) if row["product_id"] else None,
            sentiment=float(row["sentiment"]),
            issue_category=row["issue_category"],
            description=row["description"],
            created_at=now - dt.timedelta(days=int(row["days_ago"])),
        )
        for row in rows
    ]
    session.add_all(tickets)
    print(f"🎫 Loaded {len(tickets)} support tickets.")


async def _reset_sequences(session: AsyncSession) -> None:
    """Reset PostgreSQL sequences to max ID + 1 for each table."""
    sequences = [
        ("products", "products_id_seq"),
        ("campaigns", "campaigns_id_seq"),
        ("orders", "orders_id_seq"),
        ("support_tickets", "support_tickets_id_seq"),
    ]

    for table, seq in sequences:
        await session.execute(
            text(f"SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)")
        )


async def _print_summary(session: AsyncSession) -> None:
    """Print a summary of seeded data."""
    from sqlalchemy import func, select

    counts = {}
    for model, name in [
        (Product, "Products"),
        (Campaign, "Campaigns"),
        (Order, "Orders"),
        (SupportTicket, "Support Tickets"),
    ]:
        result = await session.execute(select(func.count()).select_from(model))
        counts[name] = result.scalar()

    print("\n📊 Database Summary:")
    print("-" * 30)
    for name, count in counts.items():
        print(f"   {name}: {count}")
    print("-" * 30)


if __name__ == "__main__":
    import sys

    clear_only = "--clear" in sys.argv
    asyncio.run(seed(clear_only=clear_only))
