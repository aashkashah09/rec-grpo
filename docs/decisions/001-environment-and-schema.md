# 001 — Environment data model and schema

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Environment, tasks, verifier)

## Context

Phase 1 needs a deterministic SQL-analytics environment: a seeded e-commerce database that
task ground truth can be computed from programmatically (`PROJECT_PLAN.md` §3, §1.3). The
schema must support all 8 task templates (revenue, refund rate, top-K, growth, LTV, return-rate
anomaly, ratio, and a NULL/edge case) without contortion.

## Decisions

1. **Six tables** — `customers`, `products`, `orders`, `order_items`, `refunds` (order-level
   monetary), `returns` (item-level physical units). Refunds and returns are modelled
   separately because the project distinguishes a monetary refund rate from a physical return
   rate; conflating them would make two templates ill-defined.
2. **Money as integer cents.** All monetary columns are `INTEGER` cents; conversion to USD
   dollars happens only at the verifier boundary. This eliminates floating-point drift in
   ground truth and makes the reference SQL and Python references agree exactly.
3. **ISO-8601 TEXT dates/timestamps** (`YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`). Deterministic,
   diff-friendly, and directly comparable with SQLite `strftime`.
4. **`order_items.discount_cents` is NULLABLE**, with NULL meaning "no discount recorded"
   (distinct from an explicit 0). This is what the `null_discount_edge` template exercises.
5. **NumPy `default_rng(seed)` generation.** Data is fully determined by `(config, seed)` so
   runs and artifacts are reproducible on CPU. Row counts, vocabularies, and event
   probabilities are config-driven (`configs/env.yaml`, `env.mini.yaml`).
6. **Two views, cross-checked.** The generator produces both a SQLite database and a pure-Python
   `Dataset` (typed row lists). Ground truth is computed in Python over the `Dataset`; the
   oracle runs reference SQL over the database; tests assert they agree (see ADR-003).
7. **Frozen fixture as a `.sql` seed file.** The mini test database is committed as
   `tests/fixtures/mini_db.sql` (INSERTs only; schema loaded from `schema_sql()`), not as a
   gitignored binary. Human-readable, diff-able, and stable across NumPy versions, so the
   hand-computed expected literals never drift.

## Consequences

- Ground truth is trivially auditable (integer cents, explicit formulas).
- Adding a template that needs a new relationship may require a schema migration + fixture
  update; the frozen fixture's literals must be recomputed if its rows change.
- The generator holds the full dataset in memory (fine at the single-node scale targeted).
