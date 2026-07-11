"""The 8 SQL-analytics task templates: sampling, question text, ground truth, reference SQL.

Each template exposes four things:

* ``sample_params`` — a seeded parameter draw from schema-consistent domains,
* ``render_question`` — question text that states its **exact formula** and its **expected
  answer format** (a reviewer, and the verifier, should never have to guess units),
* ``ground_truth`` — a pure-Python reference computed over an :class:`EnvIndex` (the source of
  truth), and
* ``reference_sql`` — the equivalent query the oracle agent runs through the sandbox.

The pure-Python and SQL paths are independent implementations of the same definition; tests
assert they agree, which is the environment's core correctness guarantee. Two revenue notions
recur and are deliberately given distinct names to avoid conflation:
``net_of_discounts_cents`` (gross minus item discounts) and ``net_of_refunds_cents``
(net-of-discounts minus order refunds).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from specialist_router.config import Config
from specialist_router.env.database import (
    Customer,
    Dataset,
    Order,
    OrderItem,
    Product,
    Refund,
    Return,
)
from specialist_router.env.records import AnswerType, AnswerValue, Difficulty, Task

# The substring each rendered question must contain so the expected answer format is explicit
# (asserted by tests). Keyed by answer type.
FORMAT_MARKERS: dict[AnswerType, str] = {
    AnswerType.MONEY_USD: "US dollars",
    AnswerType.RATIO: "decimal fraction",
    AnswerType.PERCENTAGE_POINTS: "percentage points",
    AnswerType.INTEGER: "integer",
    AnswerType.LIST_STR: "ordered list",
}


# --------------------------------------------------------------------------------------------
# Named revenue helpers (see module docstring: net-of-discounts vs net-of-refunds).
# --------------------------------------------------------------------------------------------


def gross_revenue_cents(items: Iterable[OrderItem]) -> int:
    """Sum of ``quantity * unit_price_cents`` (before any discount)."""
    return sum(i.quantity * i.unit_price_cents for i in items)


def net_of_discounts_cents(items: Iterable[OrderItem]) -> int:
    """Gross revenue minus item discounts, treating a NULL discount as 0."""
    return sum(i.quantity * i.unit_price_cents - (i.discount_cents or 0) for i in items)


def net_of_refunds_cents(items: Iterable[OrderItem], refunds: Iterable[Refund]) -> int:
    """Net-of-discounts revenue minus order-level refunds."""
    return net_of_discounts_cents(items) - sum(r.amount_cents for r in refunds)


def _cents_to_usd(cents: float) -> float:
    """Convert integer cents to a USD-dollar float (the money answer unit)."""
    return cents / 100.0


# --------------------------------------------------------------------------------------------
# EnvIndex: precomputed lookups so pure-Python ground truth is O(n) once, not per-task.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvIndex:
    """Precomputed indexes over a :class:`Dataset` shared by all ground-truth computations."""

    dataset: Dataset
    product_by_id: dict[int, Product]
    customer_by_id: dict[int, Customer]
    order_by_id: dict[int, Order]
    items_by_order: dict[int, list[OrderItem]]
    refunds_by_order: dict[int, list[Refund]]
    returns_by_item: dict[int, list[Return]]
    order_years: list[int]
    signup_years: list[int]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> EnvIndex:
        """Build all lookups from ``dataset`` in a single pass per table."""
        items_by_order: dict[int, list[OrderItem]] = {}
        for item in dataset.order_items:
            items_by_order.setdefault(item.order_id, []).append(item)
        refunds_by_order: dict[int, list[Refund]] = {}
        for refund in dataset.refunds:
            refunds_by_order.setdefault(refund.order_id, []).append(refund)
        returns_by_item: dict[int, list[Return]] = {}
        for ret in dataset.returns:
            returns_by_item.setdefault(ret.order_item_id, []).append(ret)
        return cls(
            dataset=dataset,
            product_by_id={p.product_id: p for p in dataset.products},
            customer_by_id={c.customer_id: c for c in dataset.customers},
            order_by_id={o.order_id: o for o in dataset.orders},
            items_by_order=items_by_order,
            refunds_by_order=refunds_by_order,
            returns_by_item=returns_by_item,
            order_years=sorted({int(o.order_ts[:4]) for o in dataset.orders}),
            signup_years=sorted({int(c.signup_date[:4]) for c in dataset.customers}),
        )

    def completed_orders(self) -> list[Order]:
        """Return all orders with status 'completed'."""
        return [o for o in self.dataset.orders if o.status == "completed"]


def _choice(rng: np.random.Generator, seq: list[int] | list[str]) -> int | str:
    """Pick one element of ``seq`` using ``rng`` (index-based for determinism)."""
    return seq[int(rng.integers(0, len(seq)))]


def _pint(params: dict[str, object], key: str) -> int:
    value = params[key]
    if not isinstance(value, int):
        raise TypeError(f"param {key!r} must be int, got {type(value).__name__}")
    return value


def _pstr(params: dict[str, object], key: str) -> str:
    value = params[key]
    if not isinstance(value, str):
        raise TypeError(f"param {key!r} must be str, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------------------------
# Template base class.
# --------------------------------------------------------------------------------------------


class TaskTemplate(ABC):
    """A parameterised task family with dual (Python + SQL) ground truth."""

    id: str
    difficulty: Difficulty
    answer_type: AnswerType

    @abstractmethod
    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        """Draw a valid parameter set from schema-consistent, non-degenerate domains."""

    @abstractmethod
    def render_question(self, params: dict[str, object]) -> str:
        """Render the question, stating the exact formula and the expected answer format."""

    @abstractmethod
    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        """Compute the reference answer in pure Python (the source of truth)."""

    @abstractmethod
    def reference_sql(self, params: dict[str, object]) -> str:
        """Return the equivalent read-only SQL the oracle agent runs through the sandbox."""


class RevenueBySegment(TaskTemplate):
    """Total net-of-discounts revenue for a segment in a year (easy)."""

    id = "revenue_by_segment"
    difficulty: Difficulty = "easy"
    answer_type = AnswerType.MONEY_USD

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        return {
            "segment": _choice(rng, config.db.segments),
            "year": _choice(rng, index.order_years),
        }

    def render_question(self, params: dict[str, object]) -> str:
        segment, year = _pstr(params, "segment"), _pint(params, "year")
        return (
            f"What was the total net revenue, in US dollars, from completed orders placed by "
            f"{segment} customers during {year}? Net revenue is the sum over order items of "
            f"(quantity x unit_price - discount), where a missing (NULL) discount counts as 0, "
            f"and only orders with status 'completed' are included. "
            f"Answer in US dollars (e.g. 1234.56)."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        segment, year = _pstr(params, "segment"), _pint(params, "year")
        seg_customers = {c.customer_id for c in index.dataset.customers if c.segment == segment}
        cents = 0
        for order in index.dataset.orders:
            if (
                order.status == "completed"
                and int(order.order_ts[:4]) == year
                and order.customer_id in seg_customers
            ):
                cents += net_of_discounts_cents(index.items_by_order.get(order.order_id, []))
        return _cents_to_usd(cents)

    def reference_sql(self, params: dict[str, object]) -> str:
        segment, year = _pstr(params, "segment"), _pint(params, "year")
        return (
            "SELECT COALESCE(SUM(oi.quantity * oi.unit_price_cents "
            "- COALESCE(oi.discount_cents, 0)), 0) / 100.0 "
            "FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id "
            "JOIN order_items oi ON oi.order_id = o.order_id "
            f"WHERE o.status = 'completed' AND c.segment = '{segment}' "
            f"AND strftime('%Y', o.order_ts) = '{year}'"
        )


class RefundRateCohort(TaskTemplate):
    """Signed percentage-point gap between two signup cohorts' refund rates (med)."""

    id = "refund_rate_cohort"
    difficulty: Difficulty = "med"
    answer_type = AnswerType.PERCENTAGE_POINTS

    def _cohort_years(self, index: EnvIndex) -> list[int]:
        """Signup years whose cohort has positive gross revenue on completed orders."""
        gross_by_year: dict[int, int] = {}
        completed = {o.order_id for o in index.completed_orders()}
        cust_year = {c.customer_id: int(c.signup_date[:4]) for c in index.dataset.customers}
        for order in index.dataset.orders:
            if order.order_id not in completed:
                continue
            year = cust_year[order.customer_id]
            gross_by_year[year] = gross_by_year.get(year, 0) + gross_revenue_cents(
                index.items_by_order.get(order.order_id, [])
            )
        return sorted(y for y, g in gross_by_year.items() if g > 0)

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        years = self._cohort_years(index)
        if len(years) < 2:
            raise ValueError("refund_rate_cohort needs >= 2 cohort years with revenue")
        first = int(_choice(rng, years))
        rest = [y for y in years if y != first]
        return {"year_a": first, "year_b": int(_choice(rng, rest))}

    def render_question(self, params: dict[str, object]) -> str:
        year_a, year_b = _pint(params, "year_a"), _pint(params, "year_b")
        return (
            f"Consider two signup cohorts: customers who signed up in {year_a} and in {year_b}. "
            f"For each cohort, define its refund rate as (total refund amount on the cohort's "
            f"completed orders) / (gross revenue of those orders), where gross revenue is the sum "
            f"of quantity x unit_price before discounts. By how many percentage points did the "
            f"{year_a} cohort's refund rate exceed the {year_b} cohort's? Report a signed number "
            f"of percentage points (e.g. 2.5 means 2.5 percentage points; it may be negative)."
        )

    def _rate(self, index: EnvIndex, year: int) -> float:
        cust = {c.customer_id for c in index.dataset.customers if int(c.signup_date[:4]) == year}
        order_ids = {
            o.order_id
            for o in index.dataset.orders
            if o.status == "completed" and o.customer_id in cust
        }
        gross = gross_revenue_cents(
            i for oid in order_ids for i in index.items_by_order.get(oid, [])
        )
        refunds = sum(
            r.amount_cents for oid in order_ids for r in index.refunds_by_order.get(oid, [])
        )
        return refunds / gross

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        year_a, year_b = _pint(params, "year_a"), _pint(params, "year_b")
        return 100.0 * (self._rate(index, year_a) - self._rate(index, year_b))

    def reference_sql(self, params: dict[str, object]) -> str:
        year_a, year_b = _pint(params, "year_a"), _pint(params, "year_b")
        return (
            "WITH cohort AS ("
            "  SELECT o.order_id AS order_id, substr(c.signup_date, 1, 4) AS yr "
            "  FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "  WHERE o.status = 'completed'"
            "), "
            "gross AS ("
            "  SELECT ch.yr AS yr, SUM(oi.quantity * oi.unit_price_cents) AS g "
            "  FROM cohort ch JOIN order_items oi ON oi.order_id = ch.order_id GROUP BY ch.yr"
            "), "
            "ref AS ("
            "  SELECT ch.yr AS yr, SUM(r.amount_cents) AS rf "
            "  FROM cohort ch JOIN refunds r ON r.order_id = ch.order_id GROUP BY ch.yr"
            ") "
            "SELECT 100.0 * ("
            f"  COALESCE((SELECT rf FROM ref WHERE yr = '{year_a}'), 0) * 1.0 "
            f"    / (SELECT g FROM gross WHERE yr = '{year_a}') "
            f"  - COALESCE((SELECT rf FROM ref WHERE yr = '{year_b}'), 0) * 1.0 "
            f"    / (SELECT g FROM gross WHERE yr = '{year_b}'))"
        )


class TopKCategories(TaskTemplate):
    """Top-K product categories by units sold in a year, with a deterministic tie-break (med)."""

    id = "topk_categories"
    difficulty: Difficulty = "med"
    answer_type = AnswerType.LIST_STR

    def _units_by_category(self, index: EnvIndex, year: int) -> dict[str, int]:
        units: dict[str, int] = {}
        for order in index.dataset.orders:
            if order.status != "completed" or int(order.order_ts[:4]) != year:
                continue
            for item in index.items_by_order.get(order.order_id, []):
                category = index.product_by_id[item.product_id].category
                units[category] = units.get(category, 0) + item.quantity
        return units

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        year = int(_choice(rng, index.order_years))
        available = len(self._units_by_category(index, year))
        k = min(int(_choice(rng, config.tasks.k_choices)), available)
        return {"k": max(1, k), "year": year}

    def render_question(self, params: dict[str, object]) -> str:
        k, year = _pint(params, "k"), _pint(params, "year")
        return (
            f"List the top {k} product categories by total units sold (sum of quantity) across "
            f"completed orders during {year}. Break ties by category name in ascending "
            f"alphabetical order. Answer as an ordered list of category names, highest first."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        k, year = _pint(params, "k"), _pint(params, "year")
        units = self._units_by_category(index, year)
        ranked = sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in ranked[:k]]

    def reference_sql(self, params: dict[str, object]) -> str:
        k, year = _pint(params, "k"), _pint(params, "year")
        return (
            "SELECT p.category "
            "FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.order_id "
            "JOIN products p ON p.product_id = oi.product_id "
            f"WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}' "
            "GROUP BY p.category "
            f"ORDER BY SUM(oi.quantity) DESC, p.category ASC LIMIT {k}"
        )


class MomGrowth(TaskTemplate):
    """Month-over-month net-revenue growth as a signed fraction (med)."""

    id = "mom_growth"
    difficulty: Difficulty = "med"
    answer_type = AnswerType.RATIO

    def _rev_by_month(self, index: EnvIndex, year: int) -> dict[int, int]:
        rev: dict[int, int] = {}
        for order in index.dataset.orders:
            if order.status != "completed" or int(order.order_ts[:4]) != year:
                continue
            month = int(order.order_ts[5:7])
            rev[month] = rev.get(month, 0) + net_of_discounts_cents(
                index.items_by_order.get(order.order_id, [])
            )
        return rev

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        for year in sorted(index.order_years, key=lambda _: int(rng.integers(0, 1_000_000))):
            rev = self._rev_by_month(index, year)
            valid = [m for m in range(1, 12) if rev.get(m, 0) > 0]
            if valid:
                m1 = int(_choice(rng, valid))
                return {"year": year, "m1": m1, "m2": m1 + 1}
        raise ValueError("mom_growth found no month with positive revenue")

    def render_question(self, params: dict[str, object]) -> str:
        year, m1, m2 = _pint(params, "year"), _pint(params, "m1"), _pint(params, "m2")
        return (
            f"Using net revenue = sum over order items of (quantity x unit_price - discount, with "
            f"NULL discount = 0) on completed orders, what was the growth from month {m1:02d} to "
            f"month {m2:02d} of {year}, computed as (revenue_month2 - revenue_month1) / "
            f"revenue_month1? Answer as a decimal fraction (e.g. 0.1234), not a percentage; it "
            f"may be negative."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        year, m1, m2 = _pint(params, "year"), _pint(params, "m1"), _pint(params, "m2")
        rev = self._rev_by_month(index, year)
        rev1, rev2 = rev.get(m1, 0), rev.get(m2, 0)
        return (rev2 - rev1) / rev1

    def reference_sql(self, params: dict[str, object]) -> str:
        year, m1, m2 = _pint(params, "year"), _pint(params, "m1"), _pint(params, "m2")
        return (
            "WITH m AS ("
            "  SELECT strftime('%m', o.order_ts) AS mo, "
            "  SUM(oi.quantity * oi.unit_price_cents - COALESCE(oi.discount_cents, 0)) AS rev "
            "  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
            f"  WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}' "
            "  GROUP BY mo"
            ") "
            f"SELECT (COALESCE((SELECT rev FROM m WHERE mo = '{m2:02d}'), 0) * 1.0 "
            f"  - COALESCE((SELECT rev FROM m WHERE mo = '{m1:02d}'), 0)) "
            f"  / COALESCE((SELECT rev FROM m WHERE mo = '{m1:02d}'), 0)"
        )


class CustomerLtv(TaskTemplate):
    """Lifetime net-of-refunds revenue for a single customer (hard, join-heavy)."""

    id = "customer_ltv"
    difficulty: Difficulty = "hard"
    answer_type = AnswerType.MONEY_USD

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        eligible = sorted({o.customer_id for o in index.completed_orders()})
        if not eligible:
            raise ValueError("customer_ltv found no customer with a completed order")
        return {"customer_id": int(_choice(rng, eligible))}

    def render_question(self, params: dict[str, object]) -> str:
        customer_id = _pint(params, "customer_id")
        return (
            f"What is the lifetime net revenue, in US dollars, of customer {customer_id}? Define "
            f"it as [sum over order items on the customer's completed orders of "
            f"(quantity x unit_price - discount, NULL = 0)] minus [total refunds on those "
            f"completed orders]. Answer in US dollars (e.g. 1234.56)."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        customer_id = _pint(params, "customer_id")
        order_ids = {
            o.order_id
            for o in index.dataset.orders
            if o.customer_id == customer_id and o.status == "completed"
        }
        items = [i for oid in order_ids for i in index.items_by_order.get(oid, [])]
        refunds = [r for oid in order_ids for r in index.refunds_by_order.get(oid, [])]
        return _cents_to_usd(net_of_refunds_cents(items, refunds))

    def reference_sql(self, params: dict[str, object]) -> str:
        customer_id = _pint(params, "customer_id")
        return (
            "SELECT ("
            "  COALESCE((SELECT SUM(oi.quantity * oi.unit_price_cents "
            "    - COALESCE(oi.discount_cents, 0)) "
            "    FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
            f"    WHERE o.customer_id = {customer_id} AND o.status = 'completed'), 0) "
            "  - COALESCE((SELECT SUM(r.amount_cents) "
            "    FROM orders o JOIN refunds r ON r.order_id = o.order_id "
            f"    WHERE o.customer_id = {customer_id} AND o.status = 'completed'), 0)"
            ") / 100.0"
        )


class ReturnRateAnomaly(TaskTemplate):
    """Product with the highest return rate above a sales threshold (hard)."""

    id = "return_rate_anomaly"
    difficulty: Difficulty = "hard"
    answer_type = AnswerType.INTEGER

    def _scope(self, index: EnvIndex, year: int) -> tuple[dict[int, int], dict[int, int]]:
        """Return (units_sold, units_returned) per product over completed orders in the year."""
        sold: dict[int, int] = {}
        returned: dict[int, int] = {}
        for order in index.dataset.orders:
            if order.status != "completed" or int(order.order_ts[:4]) != year:
                continue
            for item in index.items_by_order.get(order.order_id, []):
                sold[item.product_id] = sold.get(item.product_id, 0) + item.quantity
                for ret in index.returns_by_item.get(item.order_item_id, []):
                    returned[item.product_id] = (
                        returned.get(item.product_id, 0) + ret.quantity_returned
                    )
        return sold, returned

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        for year in sorted(index.order_years, key=lambda _: int(rng.integers(0, 1_000_000))):
            sold, _ = self._scope(index, year)
            if not sold:
                continue
            max_units = max(sold.values())
            choices = [m for m in config.tasks.min_units_choices if m <= max_units] or [1]
            return {"year": year, "min_units": int(_choice(rng, choices))}
        raise ValueError("return_rate_anomaly found no year with sales")

    def render_question(self, params: dict[str, object]) -> str:
        year, min_units = _pint(params, "year"), _pint(params, "min_units")
        return (
            f"Among products with at least {min_units} units sold (sum of quantity) on completed "
            f"orders during {year}, which product had the highest return rate, defined as "
            f"(units returned) / (units sold) over that same set of orders? Break ties by "
            f"choosing the lowest product_id. Return the product_id as a single integer."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        year, min_units = _pint(params, "year"), _pint(params, "min_units")
        sold, returned = self._scope(index, year)
        candidates = [
            (-returned.get(pid, 0) / units, pid)
            for pid, units in sold.items()
            if units >= min_units
        ]
        if not candidates:
            raise ValueError("return_rate_anomaly has no product above the threshold")
        return min(candidates)[1]

    def reference_sql(self, params: dict[str, object]) -> str:
        year, min_units = _pint(params, "year"), _pint(params, "min_units")
        return (
            "WITH sold AS ("
            "  SELECT oi.product_id AS pid, SUM(oi.quantity) AS units_sold "
            "  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
            f"  WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}' "
            "  GROUP BY oi.product_id"
            "), "
            "ret AS ("
            "  SELECT oi.product_id AS pid, SUM(rt.quantity_returned) AS units_ret "
            "  FROM orders o "
            "  JOIN order_items oi ON oi.order_id = o.order_id "
            "  JOIN returns rt ON rt.order_item_id = oi.order_item_id "
            f"  WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}' "
            "  GROUP BY oi.product_id"
            ") "
            "SELECT s.pid FROM sold s LEFT JOIN ret r ON r.pid = s.pid "
            f"WHERE s.units_sold >= {min_units} "
            "ORDER BY (COALESCE(r.units_ret, 0) * 1.0 / s.units_sold) DESC, s.pid ASC LIMIT 1"
        )


class CategoryOrderRatio(TaskTemplate):
    """Fraction of completed orders in a year containing a given category (med, multi-step)."""

    id = "category_order_ratio"
    difficulty: Difficulty = "med"
    answer_type = AnswerType.RATIO

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        years = [
            y
            for y in index.order_years
            if any(
                o.status == "completed" and int(o.order_ts[:4]) == y for o in index.dataset.orders
            )
        ]
        if not years:
            raise ValueError("category_order_ratio found no year with completed orders")
        return {"category": _choice(rng, config.db.categories), "year": int(_choice(rng, years))}

    def render_question(self, params: dict[str, object]) -> str:
        category, year = _pstr(params, "category"), _pint(params, "year")
        return (
            f"Of all completed orders placed during {year}, what fraction contained at least one "
            f"order item whose product is in the {category} category? Compute (number of such "
            f"orders) / (number of completed orders in {year}). Answer as a decimal fraction "
            f"(e.g. 0.1234), not a percentage."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        category, year = _pstr(params, "category"), _pint(params, "year")
        completed = [
            o
            for o in index.dataset.orders
            if o.status == "completed" and int(o.order_ts[:4]) == year
        ]
        denom = len(completed)
        cat_products = {p.product_id for p in index.dataset.products if p.category == category}
        matching = sum(
            1
            for o in completed
            if any(i.product_id in cat_products for i in index.items_by_order.get(o.order_id, []))
        )
        return matching / denom

    def reference_sql(self, params: dict[str, object]) -> str:
        category, year = _pstr(params, "category"), _pint(params, "year")
        return (
            "SELECT (SELECT COUNT(DISTINCT o.order_id) "
            "  FROM orders o "
            "  JOIN order_items oi ON oi.order_id = o.order_id "
            "  JOIN products p ON p.product_id = oi.product_id "
            f"  WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}' "
            f"  AND p.category = '{category}') * 1.0 "
            "/ (SELECT COUNT(*) FROM orders o "
            f"   WHERE o.status = 'completed' AND strftime('%Y', o.order_ts) = '{year}')"
        )


class NullDiscountEdge(TaskTemplate):
    """Average per-item discount in a year, treating NULL as 0, with an empty-set branch (med)."""

    id = "null_discount_edge"
    difficulty: Difficulty = "med"
    answer_type = AnswerType.MONEY_USD

    def sample_params(
        self, rng: np.random.Generator, index: EnvIndex, config: Config
    ) -> dict[str, object]:
        # Include the year before the data window so the "no items -> 0" branch is exercised.
        candidates = [min(index.order_years) - 1, *index.order_years]
        return {"year": int(_choice(rng, candidates))}

    def render_question(self, params: dict[str, object]) -> str:
        year = _pint(params, "year")
        return (
            f"What was the average discount, in US dollars, per order item on orders placed during "
            f"{year}, treating a missing (NULL) discount as 0? Average over all order items whose "
            f"order falls in {year}; if there are no such items, answer 0. Answer in US dollars "
            f"(e.g. 12.34)."
        )

    def ground_truth(self, index: EnvIndex, params: dict[str, object]) -> AnswerValue:
        year = _pint(params, "year")
        order_ids = {o.order_id for o in index.dataset.orders if int(o.order_ts[:4]) == year}
        discounts = [
            (i.discount_cents or 0) for oid in order_ids for i in index.items_by_order.get(oid, [])
        ]
        if not discounts:
            return 0.0
        return _cents_to_usd(sum(discounts) / len(discounts))

    def reference_sql(self, params: dict[str, object]) -> str:
        year = _pint(params, "year")
        return (
            "SELECT COALESCE(AVG(COALESCE(oi.discount_cents, 0)), 0) / 100.0 "
            "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
            f"WHERE strftime('%Y', o.order_ts) = '{year}'"
        )


TEMPLATES: dict[str, TaskTemplate] = {
    t.id: t
    for t in (
        RevenueBySegment(),
        RefundRateCohort(),
        TopKCategories(),
        MomGrowth(),
        CustomerLtv(),
        ReturnRateAnomaly(),
        CategoryOrderRatio(),
        NullDiscountEdge(),
    )
}


def generate_tasks(dataset: Dataset, config: Config) -> list[Task]:
    """Generate ``config.tasks.n_tasks`` tasks round-robin across the enabled templates.

    Round-robin (rather than random) selection guarantees balanced coverage of all templates,
    which matters for held-out evaluation later. Sampling uses ``seed + 1`` so task parameters
    are decoupled from the data seed yet still fully reproducible.

    Args:
        dataset: The generated dataset the tasks are grounded in.
        config: Full config (template list, counts, per-template ranges, seed).

    Returns:
        A list of :class:`Task` records with computed ground truth and reference SQL.
    """
    index = EnvIndex.from_dataset(dataset)
    rng = np.random.default_rng(config.seed + 1)
    templates = [TEMPLATES[name] for name in config.tasks.templates]
    tasks: list[Task] = []
    for i in range(config.tasks.n_tasks):
        template = templates[i % len(templates)]
        params = template.sample_params(rng, index, config)
        tasks.append(
            Task(
                task_id=f"task-{i:06d}",
                template_id=template.id,
                difficulty=template.difficulty,
                question=template.render_question(params),
                answer_type=template.answer_type,
                params=params,
                expected=template.ground_truth(index, params),
                reference_sql=template.reference_sql(params),
            )
        )
    return tasks
