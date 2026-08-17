"""Sandbox escape attempts must be rejected; safe operations must work.

Covers the two trust-critical tools: ``run_sql`` (read-only, single-SELECT, budgeted) and
``python_calc`` (AST-whitelist numeric evaluator).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from specialist_router.config import Config
from specialist_router.env.database import open_readonly
from specialist_router.env.tools import (
    CalcError,
    SqlBudgetError,
    SqlNotAllowedError,
    SqlSandbox,
    inspect_schema,
    python_calc,
)


@pytest.fixture
def sandbox(mini_db_file: Path, env_config: Config) -> SqlSandbox:
    return SqlSandbox(open_readonly(mini_db_file), env_config.tools.run_sql)


# --- run_sql: allowed reads -----------------------------------------------------------------


def test_select_returns_rows(sandbox: SqlSandbox) -> None:
    result = sandbox.execute("SELECT COUNT(*) FROM orders")
    assert int(result.scalar()) == 7


def test_with_cte_is_allowed(sandbox: SqlSandbox) -> None:
    result = sandbox.execute("WITH x AS (SELECT 1 AS n) SELECT n FROM x")
    assert int(result.scalar()) == 1


# --- run_sql: rejected writes / DDL / escapes -----------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO orders VALUES (99, 1, '2022-01-01 00:00:00', 'completed', 'web')",
        "UPDATE orders SET status = 'x'",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "CREATE TABLE hack (x INT)",
        "ALTER TABLE orders ADD COLUMN x INT",
        "ATTACH DATABASE 'other.db' AS other",
    ],
)
def test_write_and_ddl_are_rejected(sandbox: SqlSandbox, query: str) -> None:
    with pytest.raises(SqlNotAllowedError):
        sandbox.execute(query)


def test_multiple_statements_rejected(sandbox: SqlSandbox) -> None:
    with pytest.raises(SqlNotAllowedError):
        sandbox.execute("SELECT 1; DROP TABLE orders")


def test_non_select_rejected(sandbox: SqlSandbox) -> None:
    with pytest.raises(SqlNotAllowedError):
        sandbox.execute("PRAGMA table_info(orders)")


def test_reading_sqlite_master_is_denied(sandbox: SqlSandbox) -> None:
    with pytest.raises(SqlNotAllowedError):
        sandbox.execute("SELECT name FROM sqlite_master")


def test_opcode_budget_trips_on_expensive_query(mini_db_file: Path, env_config: Config) -> None:
    tight = env_config.tools.run_sql.model_copy(update={"max_ops": 50})
    sandbox = SqlSandbox(open_readonly(mini_db_file), tight)
    with pytest.raises(SqlBudgetError):
        # A cross join blows well past a 50-opcode budget.
        sandbox.execute(
            "SELECT COUNT(*) FROM order_items a, order_items b, order_items c, order_items d"
        )


def test_row_limit_truncates(mini_db_file: Path, env_config: Config) -> None:
    tight = env_config.tools.run_sql.model_copy(update={"max_rows": 3})
    sandbox = SqlSandbox(open_readonly(mini_db_file), tight)
    result = sandbox.execute("SELECT order_id FROM orders")
    assert result.truncated
    assert len(result.rows) == 3


# --- inspect_schema exposes types and foreign keys ------------------------------------------


def test_inspect_schema_includes_types_and_foreign_keys(mini_db_file: Path) -> None:
    conn = open_readonly(mini_db_file)
    text = inspect_schema(conn)
    conn.close()
    assert "Table orders:" in text
    assert "INTEGER" in text and "TEXT" in text
    assert "NULLABLE" in text  # order_items.discount_cents is nullable
    assert "customer_id -> customers(customer_id)" in text


# --- python_calc: allowed arithmetic --------------------------------------------------------


def test_calc_evaluates_arithmetic(env_config: Config) -> None:
    cfg = env_config.tools.python_calc
    assert python_calc("2 + 3 * 4", cfg) == 14.0
    assert python_calc("(10 - 4) / 2", cfg) == 3.0
    assert python_calc("min(3, 7) + max(1, 2)", cfg) == 5.0
    assert python_calc("round(3.14159, 2)", cfg) == 3.14
    assert python_calc("sqrt(9)", cfg) == 3.0


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "().__class__.__bases__",
        "open('/etc/passwd')",
        "x + 1",
        "[i for i in range(3)]",
        "lambda: 1",
        "foo(1)",
        "1 .__class__",
    ],
)
def test_calc_rejects_unsafe_expressions(env_config: Config, expr: str) -> None:
    with pytest.raises(CalcError):
        python_calc(expr, env_config.tools.python_calc)


def test_calc_rejects_huge_exponent(env_config: Config) -> None:
    with pytest.raises(CalcError):
        python_calc("10 ** 999", env_config.tools.python_calc)


def test_calc_rejects_oversized_expression(env_config: Config) -> None:
    tiny = env_config.tools.python_calc.model_copy(update={"max_nodes": 3})
    with pytest.raises(CalcError):
        python_calc("1 + 2 + 3 + 4", tiny)
