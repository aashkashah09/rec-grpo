"""Seeded e-commerce data generation and the in-memory ``Dataset`` view.

Two views of the same data are used throughout Phase 1:

* the SQLite database (what agents query through the sandbox), and
* the :class:`Dataset` — plain typed row lists that the pure-Python ground-truth reference in
  :mod:`specialist_router.env.tasks` operates on, *independently* of any SQL.

Keeping both and asserting they agree (in tests) is the project's core correctness guarantee:
a bug in the reference SQL or in the Python reference cannot pass unnoticed. Data generation is
fully determined by ``(config, seed)`` via NumPy's ``default_rng`` so runs are reproducible.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from specialist_router.config import DbConfig

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_DATE_FMT = "%Y-%m-%d"
_TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class Customer:
    """A customer row."""

    customer_id: int
    signup_date: str
    country: str
    segment: str
    marketing_channel: str


@dataclass(frozen=True, slots=True)
class Product:
    """A product row (monetary fields in integer cents)."""

    product_id: int
    name: str
    category: str
    price_cents: int
    cost_cents: int
    is_active: int


@dataclass(frozen=True, slots=True)
class Order:
    """An order row."""

    order_id: int
    customer_id: int
    order_ts: str
    status: str
    channel: str


@dataclass(frozen=True, slots=True)
class OrderItem:
    """An order-item row; ``discount_cents`` is ``None`` when no discount was recorded."""

    order_item_id: int
    order_id: int
    product_id: int
    quantity: int
    unit_price_cents: int
    discount_cents: int | None


@dataclass(frozen=True, slots=True)
class Refund:
    """An order-level monetary refund row."""

    refund_id: int
    order_id: int
    refund_ts: str
    amount_cents: int
    reason: str


@dataclass(frozen=True, slots=True)
class Return:
    """An item-level physical return row (units returned)."""

    return_id: int
    order_item_id: int
    return_ts: str
    quantity_returned: int
    reason: str


@dataclass(frozen=True, slots=True)
class Dataset:
    """The full generated dataset as typed row lists (the pure-Python view of the DB)."""

    customers: list[Customer]
    products: list[Product]
    orders: list[Order]
    order_items: list[OrderItem]
    refunds: list[Refund]
    returns: list[Return]


_REFUND_REASONS = ("defective", "not_as_described", "changed_mind", "late_delivery")
_RETURN_REASONS = ("wrong_size", "defective", "changed_mind", "damaged")


def _rand_ts(rng: np.random.Generator, start: datetime, end: datetime) -> str:
    """Return an ISO-8601 timestamp uniformly between ``start`` and ``end`` (inclusive)."""
    span = int((end - start).total_seconds())
    offset = int(rng.integers(0, span + 1)) if span > 0 else 0
    return (start + timedelta(seconds=offset)).strftime(_TS_FMT)


def build_dataset(config: DbConfig, seed: int) -> Dataset:
    """Generate a deterministic e-commerce dataset from ``config`` and ``seed``.

    Determinism (same inputs → identical rows) is what lets committed artifacts and tests be
    reproducible on CPU without shipping a binary database.

    Args:
        config: Row counts, date window, vocabularies, and event probabilities.
        seed: Seed for NumPy's ``default_rng``.

    Returns:
        A populated :class:`Dataset` with referential integrity by construction.
    """
    rng = np.random.default_rng(seed)
    start = datetime.strptime(config.date_start, _DATE_FMT)
    end = datetime.strptime(config.date_end, _DATE_FMT)

    customers = _gen_customers(rng, config, start, end)
    products = _gen_products(rng, config)
    orders, order_items = _gen_orders_and_items(rng, config, customers, products, end)
    refunds = _gen_refunds(rng, config, orders, order_items)
    returns = _gen_returns(rng, config, orders, order_items)
    return Dataset(customers, products, orders, order_items, refunds, returns)


def _gen_customers(
    rng: np.random.Generator, config: DbConfig, start: datetime, end: datetime
) -> list[Customer]:
    customers: list[Customer] = []
    for cid in range(1, config.n_customers + 1):
        signup = _rand_ts(rng, start, end)[:10]  # date portion only
        customers.append(
            Customer(
                customer_id=cid,
                signup_date=signup,
                country=str(rng.choice(config.countries)),
                segment=str(rng.choice(config.segments)),
                marketing_channel=str(rng.choice(config.marketing_channels)),
            )
        )
    return customers


def _gen_products(rng: np.random.Generator, config: DbConfig) -> list[Product]:
    products: list[Product] = []
    for pid in range(1, config.n_products + 1):
        price = int(rng.integers(config.price_cents_min, config.price_cents_max + 1))
        category = str(rng.choice(config.categories))
        products.append(
            Product(
                product_id=pid,
                name=f"{category}-{pid:05d}",
                category=category,
                price_cents=price,
                cost_cents=int(round(price * config.cost_fraction)),
                is_active=int(rng.integers(0, 2)),
            )
        )
    return products


def _pick_status(rng: np.random.Generator, config: DbConfig) -> str:
    draw = float(rng.random())
    if draw < config.cancel_prob:
        return "cancelled"
    if draw < config.cancel_prob + config.pending_prob:
        return "pending"
    return "completed"


def _pick_discount(rng: np.random.Generator, config: DbConfig, line_gross: int) -> int | None:
    """Choose a line discount: a positive amount, an explicit 0, or NULL (not recorded)."""
    draw = float(rng.random())
    if draw < config.discount_prob:
        # A positive discount strictly less than the line's gross value.
        return int(rng.integers(1, max(2, line_gross)))
    if draw < config.discount_prob + config.null_discount_prob:
        return None
    return 0


def _gen_orders_and_items(
    rng: np.random.Generator,
    config: DbConfig,
    customers: list[Customer],
    products: list[Product],
    end: datetime,
) -> tuple[list[Order], list[OrderItem]]:
    orders: list[Order] = []
    items: list[OrderItem] = []
    item_id = 0
    for oid in range(1, config.n_orders + 1):
        customer = customers[int(rng.integers(0, len(customers)))]
        signup = datetime.strptime(customer.signup_date, _DATE_FMT)
        order_ts = _rand_ts(rng, signup, end)
        orders.append(
            Order(
                order_id=oid,
                customer_id=customer.customer_id,
                order_ts=order_ts,
                status=_pick_status(rng, config),
                channel=str(rng.choice(config.channels)),
            )
        )
        n_items = int(rng.integers(1, config.max_items_per_order + 1))
        for _ in range(n_items):
            item_id += 1
            product = products[int(rng.integers(0, len(products)))]
            quantity = int(rng.integers(1, config.max_quantity + 1))
            line_gross = quantity * product.price_cents
            items.append(
                OrderItem(
                    order_item_id=item_id,
                    order_id=oid,
                    product_id=product.product_id,
                    quantity=quantity,
                    unit_price_cents=product.price_cents,
                    discount_cents=_pick_discount(rng, config, line_gross),
                )
            )
    return orders, items


def _gen_refunds(
    rng: np.random.Generator,
    config: DbConfig,
    orders: list[Order],
    items: list[OrderItem],
) -> list[Refund]:
    """Generate order-level refunds on completed orders (amount bounded by order gross)."""
    gross_by_order: dict[int, int] = {}
    for it in items:
        gross_by_order[it.order_id] = (
            gross_by_order.get(it.order_id, 0) + it.quantity * it.unit_price_cents
        )
    refunds: list[Refund] = []
    refund_id = 0
    for order in orders:
        if order.status != "completed":
            continue
        if float(rng.random()) >= config.refund_prob:
            continue
        gross = gross_by_order.get(order.order_id, 0)
        if gross <= 0:
            continue
        refund_id += 1
        order_dt = datetime.strptime(order.order_ts, _TS_FMT)
        refunds.append(
            Refund(
                refund_id=refund_id,
                order_id=order.order_id,
                refund_ts=(order_dt + timedelta(days=int(rng.integers(1, 30)))).strftime(_TS_FMT),
                amount_cents=int(rng.integers(1, gross + 1)),
                reason=str(rng.choice(_REFUND_REASONS)),
            )
        )
    return refunds


def _gen_returns(
    rng: np.random.Generator,
    config: DbConfig,
    orders: list[Order],
    items: list[OrderItem],
) -> list[Return]:
    """Generate item-level returns on items belonging to completed orders."""
    completed = {o.order_id for o in orders if o.status == "completed"}
    returns: list[Return] = []
    return_id = 0
    order_ts_by_id = {o.order_id: o.order_ts for o in orders}
    for it in items:
        if it.order_id not in completed:
            continue
        if float(rng.random()) >= config.return_prob:
            continue
        return_id += 1
        order_dt = datetime.strptime(order_ts_by_id[it.order_id], _TS_FMT)
        returns.append(
            Return(
                return_id=return_id,
                order_item_id=it.order_item_id,
                return_ts=(order_dt + timedelta(days=int(rng.integers(1, 30)))).strftime(_TS_FMT),
                quantity_returned=int(rng.integers(1, it.quantity + 1)),
                reason=str(rng.choice(_RETURN_REASONS)),
            )
        )
    return returns


def schema_sql() -> str:
    """Return the DDL text from ``schema.sql`` (single source of the schema)."""
    return _SCHEMA_PATH.read_text()


def _insert_dataset(conn: sqlite3.Connection, dataset: Dataset) -> None:
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (c.customer_id, c.signup_date, c.country, c.segment, c.marketing_channel)
            for c in dataset.customers
        ],
    )
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
        [
            (p.product_id, p.name, p.category, p.price_cents, p.cost_cents, p.is_active)
            for p in dataset.products
        ],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        [(o.order_id, o.customer_id, o.order_ts, o.status, o.channel) for o in dataset.orders],
    )
    conn.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                i.order_item_id,
                i.order_id,
                i.product_id,
                i.quantity,
                i.unit_price_cents,
                i.discount_cents,
            )
            for i in dataset.order_items
        ],
    )
    conn.executemany(
        "INSERT INTO refunds VALUES (?, ?, ?, ?, ?)",
        [(r.refund_id, r.order_id, r.refund_ts, r.amount_cents, r.reason) for r in dataset.refunds],
    )
    conn.executemany(
        "INSERT INTO returns VALUES (?, ?, ?, ?, ?)",
        [
            (r.return_id, r.order_item_id, r.return_ts, r.quantity_returned, r.reason)
            for r in dataset.returns
        ],
    )


def write_sqlite_file(dataset: Dataset, path: str | Path) -> None:
    """Write ``dataset`` to a SQLite file at ``path`` (creating schema + rows, FKs enforced).

    A file (not in-memory) database is used so tools can open it with ``mode=ro`` for an
    OS-level read-only guarantee.
    """
    target = Path(path)
    if target.exists():
        target.unlink()
    conn = sqlite3.connect(str(target))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema_sql())
        _insert_dataset(conn, dataset)
        conn.commit()
    finally:
        conn.close()


def connect_populated_memory(dataset: Dataset) -> sqlite3.Connection:
    """Return a writable in-memory connection populated from ``dataset`` (for tests/loading)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql())
    _insert_dataset(conn, dataset)
    conn.commit()
    return conn


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite file as a read-only connection (``mode=ro``)."""
    uri = f"file:{Path(path).resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_dataset_from_connection(conn: sqlite3.Connection) -> Dataset:
    """Read all tables from ``conn`` into a :class:`Dataset` (the SQL → Python bridge).

    Used to obtain the pure-Python view for the frozen ``mini_db.sql`` fixture, which has no
    generator-produced :class:`Dataset` of its own.
    """
    customers = [
        Customer(*row) for row in conn.execute("SELECT * FROM customers ORDER BY customer_id")
    ]
    products = [Product(*row) for row in conn.execute("SELECT * FROM products ORDER BY product_id")]
    orders = [Order(*row) for row in conn.execute("SELECT * FROM orders ORDER BY order_id")]
    order_items = [
        OrderItem(*row) for row in conn.execute("SELECT * FROM order_items ORDER BY order_item_id")
    ]
    refunds = [Refund(*row) for row in conn.execute("SELECT * FROM refunds ORDER BY refund_id")]
    returns = [Return(*row) for row in conn.execute("SELECT * FROM returns ORDER BY return_id")]
    return Dataset(customers, products, orders, order_items, refunds, returns)
