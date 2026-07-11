# Specialist + Router

**RL-trained SQL agent behind an OPE-validated learned cascade.**

A GRPO-trained small language model (the "specialist") for SQL analytics tasks, deployed
behind a contextual-bandit router that chooses per query between the specialist and a
frontier API — with propensity-logged decisions and off-policy evaluation (IPS/SNIPS/DR)
used to select the routing policy, validated by replay A/B.

> **Status: Phase 0 — scaffolding.** Only the repository skeleton, tooling, and CI exist.
> There are no results yet; every metric in this README will link to a committed artifact
> when it lands. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full plan.

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

_Later phases (Environment/tasks/verifier, Router/OPE, GRPO training, Integration, Report)
are described in `PROJECT_PLAN.md` and will be filled in here as they complete._
