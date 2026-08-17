"""Data-generation invariants: determinism, referential integrity, counts, money/date types."""

from __future__ import annotations

from specialist_router.config import Config
from specialist_router.env.database import (
    build_dataset,
    connect_populated_memory,
)


def test_generation_is_deterministic(env_config: Config) -> None:
    a = build_dataset(env_config.db, env_config.seed)
    b = build_dataset(env_config.db, env_config.seed)
    assert a == b


def test_different_seed_changes_data(env_config: Config) -> None:
    a = build_dataset(env_config.db, env_config.seed)
    b = build_dataset(env_config.db, env_config.seed + 1)
    assert a.orders != b.orders


def test_row_counts_match_config(env_config: Config) -> None:
    ds = build_dataset(env_config.db, env_config.seed)
    assert len(ds.customers) == env_config.db.n_customers
    assert len(ds.products) == env_config.db.n_products
    assert len(ds.orders) == env_config.db.n_orders
    assert len(ds.order_items) >= len(ds.orders)  # >= 1 item per order


def test_referential_integrity_holds(env_config: Config) -> None:
    ds = build_dataset(env_config.db, env_config.seed)
    conn = connect_populated_memory(ds)
    try:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()
    assert violations == []


def test_money_and_discount_invariants(env_config: Config) -> None:
    ds = build_dataset(env_config.db, env_config.seed)
    assert all(isinstance(p.price_cents, int) and p.price_cents > 0 for p in ds.products)
    for item in ds.order_items:
        assert item.discount_cents is None or item.discount_cents >= 0
        assert item.quantity >= 1


def test_dates_are_iso_and_orders_after_signup(env_config: Config) -> None:
    ds = build_dataset(env_config.db, env_config.seed)
    signup = {c.customer_id: c.signup_date for c in ds.customers}
    for order in ds.orders:
        assert len(order.order_ts) == 19  # 'YYYY-MM-DD HH:MM:SS'
        assert order.order_ts[:10] >= signup[order.customer_id]
