import asyncio
import datetime as dt
from decimal import Decimal

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db.models import Campaign, Order, Product, SupportTicket
from db.session import async_session_factory


async def seed() -> None:
    async with async_session_factory() as session:
        await _clear_tables(session)
        await _seed_products(session)
        await _seed_orders(session)
        await _seed_campaigns(session)
        await _seed_support_tickets(session)
        await session.commit()
        print("Database seeded with mock data.")


async def _clear_tables(session: AsyncSession) -> None:
    for model in (SupportTicket, Order, Campaign, Product):
        await session.execute(delete(model))
    await session.commit()


async def _seed_products(session: AsyncSession) -> None:
    products = [
        Product(
            name="EcoWater Bottle",
            category="Home & Kitchen",
            price=Decimal("24.99"),
            stock_qty=120,
            low_stock_threshold=30,
        ),
        Product(
            name="LumaSmart Bulb",
            category="Electronics",
            price=Decimal("14.99"),
            stock_qty=40,
            low_stock_threshold=20,
        ),
        Product(
            name="Flexifit Yoga Mat",
            category="Fitness",
            price=Decimal("49.99"),
            stock_qty=10,
            low_stock_threshold=15,
        ),
    ]
    session.add_all(products)


async def _seed_orders(session: AsyncSession) -> None:
    orders = [
        Order(product_id=1, qty=10, revenue=Decimal("249.90"), region="US", channel="Shopify"),
        Order(product_id=1, qty=4, revenue=Decimal("99.96"), region="EU", channel="Amazon"),
        Order(product_id=2, qty=20, revenue=Decimal("299.80"), region="US", channel="Amazon"),
        Order(product_id=3, qty=5, revenue=Decimal("249.95"), region="US", channel="Shopify"),
    ]
    session.add_all(orders)


async def _seed_campaigns(session: AsyncSession) -> None:
    campaigns = [
        Campaign(
            name="Spring Hydration Push",
            budget=Decimal("1500"),
            spend=Decimal("450"),
            clicks=900,
            conversions=80,
            status="active",
        ),
        Campaign(
            name="Smart Home Flash Sale",
            budget=Decimal("2000"),
            spend=Decimal("1200"),
            clicks=1500,
            conversions=120,
            status="active",
        ),
        Campaign(
            name="Fitness Clearance",
            budget=Decimal("800"),
            spend=Decimal("600"),
            clicks=500,
            conversions=30,
            status="paused",
        ),
    ]
    session.add_all(campaigns)


async def _seed_support_tickets(session: AsyncSession) -> None:
    tickets = [
        SupportTicket(
            product_id=1,
            sentiment=0.2,
            issue_category="shipping",
            description="Delayed delivery for the water bottle.",
        ),
        SupportTicket(
            product_id=3,
            sentiment=0.3,
            issue_category="quality",
            description="Yoga mat edges fraying after a week.",
        ),
    ]
    session.add_all(tickets)


if __name__ == "__main__":
    asyncio.run(seed())
