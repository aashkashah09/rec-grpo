"""Ground-truth unit tests for every template on the frozen mini fixture.

Two independent checks per template:

1. the pure-Python ``ground_truth`` equals a **hand-computed literal** (values derived by hand
   from ``tests/fixtures/mini_db.sql``), and
2. the ``reference_sql`` result equals the Python ground truth (the dual cross-check).

If either the SQL or the Python reference had a bug, at least one check would fail.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from specialist_router.env.records import AnswerType, AnswerValue
from specialist_router.env.tasks import TEMPLATES, EnvIndex

# (template_id, params, hand-computed expected value on the frozen fixture).
CASES: list[tuple[str, dict[str, object], AnswerValue]] = [
    ("revenue_by_segment", {"segment": "consumer", "year": 2022}, 445.00),
    ("refund_rate_cohort", {"year_a": 2022, "year_b": 2023}, 100.0 * (3000 / 56000 - 6000 / 67000)),
    ("topk_categories", {"k": 2, "year": 2022}, ["electronics", "apparel"]),
    ("mom_growth", {"year": 2022, "m1": 3, "m2": 4}, (20000 - 24500) / 24500),
    ("customer_ltv", {"customer_id": 1}, 424.00),
    ("return_rate_anomaly", {"year": 2022, "min_units": 1}, 1),
    ("category_order_ratio", {"category": "apparel", "year": 2022}, 1 / 3),
    ("null_discount_edge", {"year": 2023}, 7.50),
]


def _run_reference_sql(conn: sqlite3.Connection, sql: str, answer_type: AnswerType) -> AnswerValue:
    """Execute reference SQL directly and parse the result to a typed answer value."""
    rows = conn.execute(sql).fetchall()
    if answer_type is AnswerType.LIST_STR:
        return [str(r[0]) for r in rows]
    value = rows[0][0]
    if answer_type is AnswerType.INTEGER:
        return int(value)
    return float(value) if value is not None else 0.0


@pytest.mark.parametrize("template_id, params, expected", CASES, ids=[c[0] for c in CASES])
def test_ground_truth_matches_hand_computed_literal(
    template_id: str, params: dict[str, object], expected: AnswerValue, mini_index: EnvIndex
) -> None:
    template = TEMPLATES[template_id]
    truth = template.ground_truth(mini_index, params)
    if isinstance(expected, list):
        assert truth == expected
    else:
        assert isinstance(truth, (int, float))
        assert math.isclose(float(truth), float(expected), rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("template_id, params, expected", CASES, ids=[c[0] for c in CASES])
def test_reference_sql_agrees_with_python(
    template_id: str,
    params: dict[str, object],
    expected: AnswerValue,
    mini_index: EnvIndex,
    mini_conn: sqlite3.Connection,
) -> None:
    template = TEMPLATES[template_id]
    truth = template.ground_truth(mini_index, params)
    sql_value = _run_reference_sql(mini_conn, template.reference_sql(params), template.answer_type)
    if isinstance(truth, list):
        assert sql_value == truth
    else:
        assert isinstance(sql_value, (int, float))
        assert math.isclose(float(sql_value), float(truth), rel_tol=1e-9, abs_tol=1e-9)


def test_null_discount_empty_year_returns_zero(
    mini_index: EnvIndex, mini_conn: sqlite3.Connection
) -> None:
    """A year with no items exercises the empty-set branch (answer 0), in both Python and SQL."""
    template = TEMPLATES["null_discount_edge"]
    params = {"year": 2019}
    assert template.ground_truth(mini_index, params) == 0.0
    sql_value = _run_reference_sql(mini_conn, template.reference_sql(params), template.answer_type)
    assert sql_value == 0.0
