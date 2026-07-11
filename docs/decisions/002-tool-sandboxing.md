# 002 — Tool sandboxing (run_sql and python_calc)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Environment, tasks, verifier)

## Context

`CLAUDE.md` hard rule #5 requires that `run_sql` use a read-only connection with a statement
allow-list and row/time limits, and that `python_calc` never `eval`/`exec` model text — it must
use a restricted numeric-expression parser. A model's output is untrusted input.

## Decisions

**`run_sql` — layered defenses (no single mechanism load-bearing):**

1. OS-level read-only connection: `sqlite3.connect("file:...?mode=ro", uri=True)`.
2. `PRAGMA query_only = ON` on the connection.
3. A SQLite **authorizer** allowing only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`,
   `SQLITE_RECURSIVE`; it denies writes, DDL, ATTACH, PRAGMA-writes, and reads of
   `sqlite_master` (schema is exposed only via `inspect_schema`).
4. A statement gate: exactly one statement, starting with `SELECT`/`WITH`.
5. A **row limit** (`max_rows`) and an **opcode budget** via `set_progress_handler`
   (a runaway query aborts → typed `SqlBudgetError`).

**`python_calc` — AST whitelist:** parse with `ast.parse(mode="eval")` and walk a tiny
whitelist: number literals, `+ - * / // % **`, unary `±`, and the functions
`abs/round/min/max/sqrt/floor/ceil`. Names, attributes, subscripts, comprehensions, lambdas,
and non-whitelisted calls are rejected, which blocks dunder/attribute traversal escapes. A
node-count cap and an exponent cap bound resource use.

**`inspect_schema`** returns columns **with types and foreign-key relationships** (via
`PRAGMA table_info` / `foreign_key_list` on a separate, non-authorized read-only connection),
so agents never need `sqlite_master` access through `run_sql`.

All rejections raise typed exceptions (`SqlNotAllowedError`, `SqlBudgetError`, `CalcError`);
nothing fails silently.

## Consequences

- The four run_sql defenses are independently tested with explicit escape attempts
  (DDL/DML/ATTACH/PRAGMA/multi-statement/`sqlite_master`), plus budget and row-limit tests.
- The calc whitelist is conservative; if a future template needs another math function it must
  be added to the whitelist explicitly (and tested), which is the intended friction.
