"""Hard-sandboxed tools exposed to agents: ``inspect_schema``, ``run_sql``, ``python_calc``.

These are trust-critical (``CONVENTIONS.md`` hard rule #5): a model's output must never be able to
mutate the database, read the filesystem, or execute arbitrary Python. Defenses are layered so
no single mechanism is load-bearing:

* ``run_sql`` — OS-level read-only connection (``mode=ro``) + ``PRAGMA query_only`` + a SQLite
  authorizer that allows only SELECT/READ + a single-statement gate + row and opcode budgets.
* ``python_calc`` — an AST walk over a tiny numeric whitelist; ``eval``/``exec`` are never used
  on model text.

All rejections raise typed exceptions; nothing fails silently.
"""

from __future__ import annotations

import ast
import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from specialist_router.config import PythonCalcConfig, RunSqlConfig
from specialist_router.env.database import open_readonly


class SandboxError(Exception):
    """Base class for all sandbox rejections."""


class SqlNotAllowedError(SandboxError):
    """Raised when a SQL statement is disallowed (non-SELECT, DDL/DML, multi-statement, ...)."""


class SqlBudgetError(SandboxError):
    """Raised when a SQL query exceeds the row or opcode budget."""


class CalcError(SandboxError):
    """Raised when ``python_calc`` receives a non-numeric or disallowed expression."""


# --------------------------------------------------------------------------------------------
# python_calc: AST-whitelist numeric evaluator (never eval/exec).
# --------------------------------------------------------------------------------------------

_CALC_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}

_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def python_calc(expr: str, config: PythonCalcConfig) -> float:
    """Evaluate a restricted numeric expression and return the result as a float.

    Only number literals, ``+ - * / // % **``, unary ``±``, and the whitelisted functions in
    :data:`_CALC_FUNCS` are permitted. Names, attributes, subscripts, comprehensions, lambdas,
    and non-whitelisted calls are rejected — so no attribute traversal (e.g. dunder escapes)
    or arbitrary execution is possible.

    Args:
        expr: The expression text.
        config: Node-count and exponent caps guarding against resource blowups.

    Returns:
        The numeric result as a float.

    Raises:
        CalcError: If the expression is unparseable, disallowed, or exceeds a cap.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"unparseable expression: {exc}") from exc

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > config.max_nodes:
        raise CalcError(f"expression too large ({node_count} nodes > {config.max_nodes})")

    result = _eval_calc(tree.body, config)
    return float(result)


def _eval_calc(node: ast.AST, config: PythonCalcConfig) -> float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalcError(f"non-numeric constant: {value!r}")
        # Preserve int vs float so e.g. round(x, 2) receives an integer ndigits, not 2.0.
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_calc(node.operand, config)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalcError(f"operator not allowed: {type(node.op).__name__}")
        left = _eval_calc(node.left, config)
        right = _eval_calc(node.right, config)
        if isinstance(node.op, ast.Pow) and abs(right) > config.max_exponent:
            raise CalcError(f"exponent {right} exceeds cap {config.max_exponent}")
        return op(left, right)
    if isinstance(node, ast.Call):
        return _eval_call(node, config)
    raise CalcError(f"expression node not allowed: {type(node).__name__}")


def _eval_call(node: ast.Call, config: PythonCalcConfig) -> float:
    if not isinstance(node.func, ast.Name) or node.func.id not in _CALC_FUNCS:
        raise CalcError("only whitelisted numeric functions may be called")
    if node.keywords:
        raise CalcError("keyword arguments are not allowed")
    args = [_eval_calc(arg, config) for arg in node.args]
    return float(_CALC_FUNCS[node.func.id](*args))


# --------------------------------------------------------------------------------------------
# run_sql: read-only, authorizer-gated, budgeted SELECT execution.
# --------------------------------------------------------------------------------------------

_ALLOWED_SQL_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)


def _authorizer(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
    """Allow only read operations; deny writes/DDL/ATTACH/PRAGMA and reads of ``sqlite_master``."""
    if action == sqlite3.SQLITE_READ and arg1 == "sqlite_master":
        return sqlite3.SQLITE_DENY
    if action in _ALLOWED_SQL_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _assert_single_select(query: str) -> None:
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise SqlNotAllowedError("empty query")
    if ";" in stripped:
        raise SqlNotAllowedError("multiple statements are not allowed")
    first = stripped.split(None, 1)[0].lower()
    if first not in ("select", "with"):
        raise SqlNotAllowedError(f"only SELECT/WITH queries are allowed (got '{first}')")


@dataclass(frozen=True, slots=True)
class SqlResult:
    """The typed result of a sandboxed query."""

    columns: list[str]
    rows: list[tuple[object, ...]]
    truncated: bool

    def scalar(self) -> object:
        """Return the single value of a 1x1 result (raises if the shape is not 1x1)."""
        if len(self.rows) != 1 or len(self.rows[0]) != 1:
            raise SqlBudgetError(f"expected a 1x1 result, got {len(self.rows)} rows")
        return self.rows[0][0]

    def as_text(self) -> str:
        """Render the result as compact text for a trajectory/model-facing tool output."""
        header = " | ".join(self.columns)
        body = "\n".join(
            " | ".join("NULL" if v is None else str(v) for v in row) for row in self.rows
        )
        note = "\n(truncated)" if self.truncated else ""
        return f"{header}\n{body}{note}" if body else f"{header}\n(no rows){note}"


class SqlSandbox:
    """Wraps a read-only connection and runs single-SELECT queries under strict budgets."""

    def __init__(self, conn: sqlite3.Connection, config: RunSqlConfig) -> None:
        """Attach the authorizer and ``query_only`` guard to ``conn``."""
        self._conn = conn
        self._config = config
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer)

    def execute(self, query: str) -> SqlResult:
        """Run ``query`` (a single SELECT/WITH) and return a budgeted :class:`SqlResult`.

        Raises:
            SqlNotAllowedError: If the statement is not an allowed single SELECT/WITH.
            SqlBudgetError: If the query is aborted by the opcode budget.
        """
        _assert_single_select(query)
        budget = _OpcodeBudget(self._config.max_ops)
        self._conn.set_progress_handler(budget, 1000)
        try:
            cursor = self._conn.execute(query)
            fetched = cursor.fetchmany(self._config.max_rows + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        except sqlite3.Error as exc:
            # Authorizer denials raise sqlite3.DatabaseError; the opcode-budget abort raises
            # OperationalError. Both are sqlite3.Error subclasses.
            if budget.tripped:
                raise SqlBudgetError(
                    f"query exceeded opcode budget ({self._config.max_ops})"
                ) from exc
            raise SqlNotAllowedError(f"query rejected: {exc}") from exc
        finally:
            self._conn.set_progress_handler(None, 0)
        truncated = len(fetched) > self._config.max_rows
        rows = [tuple(r) for r in fetched[: self._config.max_rows]]
        return SqlResult(columns=columns, rows=rows, truncated=truncated)

    def close(self) -> None:
        """Close the underlying read-only connection."""
        self._conn.close()


@dataclass
class _OpcodeBudget:
    """A SQLite progress-handler callback that aborts once the opcode budget is exhausted."""

    max_ops: int
    _steps: int = field(default=0)
    tripped: bool = field(default=False)

    def __call__(self) -> int:
        self._steps += 1000
        if self._steps > self.max_ops:
            self.tripped = True
            return 1
        return 0


def inspect_schema(conn: sqlite3.Connection) -> str:
    """Return a human-readable schema description: columns with types and FK relationships.

    A separate unrestricted (but still read-only) connection is used here — schema
    introspection is a first-class tool, distinct from the query sandbox, so agents never need
    to read ``sqlite_master`` through ``run_sql``.

    Args:
        conn: A connection *without* the query sandbox's authorizer (PRAGMA access needed).

    Returns:
        A multi-line description of every table's columns and foreign keys.
    """
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    blocks: list[str] = []
    for (table,) in table_rows:
        blocks.append(_describe_table(conn, str(table)))
    return "\n\n".join(blocks)


def _describe_table(conn: sqlite3.Connection, table: str) -> str:
    columns: list[str] = []
    for _cid, name, col_type, notnull, _default, pk in conn.execute(f"PRAGMA table_info({table})"):
        flags = []
        if pk:
            flags.append("PRIMARY KEY")
        if notnull:
            flags.append("NOT NULL")
        else:
            flags.append("NULLABLE")
        columns.append(f"    {name} {col_type} {' '.join(flags)}".rstrip())

    fk_lines: list[str] = []
    for row in conn.execute(f"PRAGMA foreign_key_list({table})"):
        # row = (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
        _id, _seq, ref_table, from_col, to_col, *_ = row
        fk_lines.append(f"    {from_col} -> {ref_table}({to_col})")

    fk_block = "\n".join(fk_lines) if fk_lines else "    (none)"
    return f"Table {table}:\n  columns:\n" + "\n".join(columns) + "\n  foreign_keys:\n" + fk_block


class ToolContext:
    """Bundles the three tools over one database file for use by the episode loop.

    Owns two read-only connections: an unrestricted one for :func:`inspect_schema` and a
    sandboxed one for :meth:`run_sql`.
    """

    def __init__(
        self, db_path: str | Path, run_sql_config: RunSqlConfig, calc_config: PythonCalcConfig
    ) -> None:
        """Open read-only connections and configure the SQL sandbox."""
        self._schema_conn = open_readonly(db_path)
        self._sql = SqlSandbox(open_readonly(db_path), run_sql_config)
        self._calc_config = calc_config

    def inspect_schema(self) -> str:
        """Return the schema description (columns with types and foreign keys)."""
        return inspect_schema(self._schema_conn)

    def run_sql(self, query: str) -> SqlResult:
        """Execute a sandboxed SELECT/WITH query."""
        return self._sql.execute(query)

    def python_calc(self, expr: str) -> float:
        """Evaluate a restricted numeric expression."""
        return python_calc(expr, self._calc_config)

    def close(self) -> None:
        """Close the underlying connections."""
        self._schema_conn.close()
        self._sql.close()
