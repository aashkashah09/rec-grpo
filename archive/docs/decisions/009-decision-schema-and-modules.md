# 009 — RouterDecision schema, version bump, and new Phase-2 modules

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Router + serving + logging + OPE)

## Context

Propensity-logged decisions are the trust-critical artifact of Phase 2: OPE and replay reproduce
from the committed JSONL alone. `CONVENTIONS.md` requires a versioned schema module and an ADR for
modules beyond `PROJECT_PLAN.md` §2.

## Decisions

1. **`RouterDecision` added to `env/records.py`; `SCHEMA_VERSION` 1 → 2.** The record carries
   everything an estimator or replay needs: the context (`feature_vector` + `feature_names` +
   `feature_dim`), the acting policy identity and a hash of its parameters, the action and its
   **logging propensity** (constrained `> 0`), the decomposed reward, and — so a replay can assert
   scale parity and any re-scoring reproduces — the reward weights and reference scales
   (`reward_lambda`, `reward_mu`, `cost_ref_usd`, `latency_ref_s`). The bump is backwards-safe:
   Phase-1 records are unchanged in shape and every committed artifact self-describes its
   `schema_version`, so v1 files remain readable.
2. **`Arm` is a closed literal** (`"local" | "api"`), so policies, the logger, and OPE dispatch
   exhaustively over the arm set.
3. **New modules (beyond the planned layout), recorded here:**
   `router/logger.py` (planned) also hosts `LoggedDataset`, the single dense numpy view both
   policies and estimators consume; `analysis/pipeline.py` (traffic→OPE→replay→breakage
   orchestration, so the pipeline is integration-testable); `analysis/report_tables.py` (pure
   Markdown formatters) and `analysis/plots.py` (matplotlib figures, optional `analysis` extra).
   `router/{features,policies,reward}.py`, `ope/{estimators,ci,replay}.py`, `serving/{app,clients}`
   are all in the planned layout. `ope/simulator.py` (known-truth tabular bandit for the property
   tests and breakage study) is new and recorded here.
4. **Deferred Phase-1 item landed here.** `run_episode` now catches an agent raising mid-episode and
   records `stop_reason="error"` (never silently), with tests — closing the gap where the schema had
   the branch but no agent could reach it.

## Consequences

- One versioned schema governs all JSONL; the logger validates `propensity > 0` on write and read.
- `configs/{router,ope,serving}.yaml` move from placeholder `schema_version: 0` to real, typed
  configs loaded through `config.py`; the `serving`/`analysis`/`ml` optional extras are pinned in
  `pyproject.toml` and the lock, but none are installed in CI (which stays numpy/pydantic/pyyaml).
