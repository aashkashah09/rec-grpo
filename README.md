# Specialist + Router

**RL-trained SQL agent behind an OPE-validated learned cascade.**

A GRPO-trained small language model (the "specialist") for SQL analytics tasks, deployed
behind a contextual-bandit router that chooses per query between the specialist and a
frontier API — with propensity-logged decisions and off-policy evaluation (IPS/SNIPS/DR)
used to select the routing policy, validated by replay A/B.

> **Status: Phase 1 — environment, tasks, verifier.** The deterministic SQL-analytics
> environment is implemented and tested. No model results yet; every metric in this README
> will link to a committed artifact when it lands. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
make setup    # uv sync + install pre-commit hooks
make lint     # ruff check + ruff format --check + mypy --strict src
make test     # pytest (CPU-only; gpu/external markers skipped)
```

## Repository map

```
configs/                 # per-phase YAML configs (env, router, ope, grpo, serving)
src/specialist_router/
  env/                   # Phase 1: seeded SQLite DB, tasks, sandboxed tools, verifier
  agents/                # Phase 2/4: local specialist + frontier-API agents
  router/                # Phase 2: featurizer, bandit policies, reward, propensity logger
  ope/                   # Phase 2: IPS/SNIPS/DR estimators, bootstrap CIs, replay A/B
  training/              # Phase 3: GRPO wiring, rollout, optional SFT warmup (GPU only)
  serving/               # Phase 2: FastAPI /solve endpoint
  evaluation/            # Phase 2/4: held-out success harness, frontier curve
  analysis/              # generates all report tables/figures from artifacts
scripts/                 # thin CLI entrypoints (per pipeline stage)
tests/                   # unit / property / integration
artifacts/               # committed small results (append-only)
docs/                    # architecture, decisions (ADRs), report
```

## Phases

Work proceeds one phase at a time (see [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §3). Each phase
ends with tests green, CI green, and a decision record for any spec deviation.

### Phase 0 — Scaffolding ✅

Repository skeleton, `uv`-managed `pyproject.toml`, ruff + mypy (strict) + pytest, a
pre-commit config, a CPU-only CI workflow, the `Makefile` repro targets, `.env.example`
(key names only), and empty-but-documented packages. `make test` and `make lint` pass on
the empty skeleton. No experiment code and no results — those begin in Phase 1.

### Phase 1 — Environment, tasks, verifier ✅

A seeded e-commerce SQLite database (6 tables, money as integer cents, ISO dates), 8
procedurally-generated task templates with dual ground truth, a hard-sandboxed tool layer, a
multi-turn episode loop, and a deterministic value-comparison verifier.

- **Database** ([`env/database.py`](src/specialist_router/env/database.py),
  [`schema.sql`](src/specialist_router/env/schema.sql)) — `customers`, `products`, `orders`,
  `order_items`, `refunds`, `returns`. Fully determined by `(config, seed)` via NumPy.
- **8 task templates** ([`env/tasks.py`](src/specialist_router/env/tasks.py)) —
  `revenue_by_segment`, `refund_rate_cohort`, `topk_categories`, `mom_growth`,
  `customer_ltv`, `return_rate_anomaly`, `category_order_ratio`, `null_discount_edge`. Each
  question states its **exact formula** and **expected answer format**. Ground truth is
  computed in pure Python **and** as reference SQL; tests assert they agree.
- **Sandboxed tools** ([`env/tools.py`](src/specialist_router/env/tools.py)) — `inspect_schema`
  (columns with types + foreign keys), read-only `run_sql` (authorizer + query_only + single
  statement + row/opcode budgets), and `python_calc` (AST-whitelist; never `eval`/`exec`).
- **Verifier** ([`env/verifier.py`](src/specialist_router/env/verifier.py)) — typed value
  comparison with per-type tolerance (money to $0.01, ratios/pp relative); rates are decimal
  fractions, never percentages. Verdicts are `{correct, reason, extracted, expected, ...}`.
- **Episode loop + reference agents** — max 12 turns and a tool budget enforced centrally;
  a scripted `OracleAgent` scores **100%** and a `TrivialAgent` scores **0%** on the mini
  environment (see `tests/integration/`).

Generate a task set with ground truth:

```bash
make generate-tasks                 # mini profile -> build/tasks.jsonl + build/env.sqlite
# or a full run:
uv run python scripts/generate_tasks.py --config configs/env.yaml --seed 42 \
    --out build/tasks.jsonl --db build/env.sqlite
```

A committed 16-task sample lives at
[`artifacts/samples/phase1_tasks_sample.jsonl`](artifacts/samples/phase1_tasks_sample.jsonl).
Design rationale is in [`docs/decisions/`](docs/decisions/) (ADRs 001–004).

_Later phases (Router/OPE, GRPO training, Integration, Report) are described in
`PROJECT_PLAN.md` and will be filled in here as they complete._
