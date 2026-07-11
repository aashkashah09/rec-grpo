# 012 — CPU dry-run strategy, the `training` marker, and the training extra

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 3 (GRPO specialist training)

## Context

Phase 3 introduces GPU-only code (TRL/torch/peft/vLLM/wandb). `CONVENTIONS.md` requires that no test call
external APIs or need a GPU by default, that CI stay CPU-only, and (ADR-000) that heavy ML deps stay
out of the CI lock. The user also required a CPU dry-run that exercises the full pipeline
(rollout→verify→reward→trainer step) to validate the wiring before renting hardware.

## Decision (confirmed with the user)

1. **Dry-run = mock-in-CI, tiny-real opt-in.**
   - The **CI-default** dry-run (`training/dry_run.py`, `scripts/grpo_dryrun.py`, `make grpo-dryrun`)
     uses **no** torch/trl/vLLM. Generation is a scripted, task-aware mock; everything downstream is
     real — the sandboxed episode loop, the verifier, the training reward, the per-episode reward
     cache, and the GRPO group-relative advantage. A `MockTrainer` asserts the exact input contract a
     real `GRPOTrainer` step consumes (aligned `completion_ids`/`mask`/`logprobs`, non-empty prompts,
     one advantage per episode). This runs in CI with zero heavy deps.
   - The **opt-in real** check (`tests/integration/test_grpo_trl_wiring.py`, marked `training`)
     constructs a real TRL `GRPOConfig` from `configs/grpo.yaml` to catch TRL parameter-name drift.
     It is CPU-safe (no model download / GPU) and guarded by `importorskip`.
2. **New `training` pytest marker**, joining `gpu`/`external`. A conftest hook skips all three unless
   the marker is explicitly requested with `-m <marker>`, so they are genuinely "skipped by default
   and in CI" (previously the markers were declared but unenforced).
3. **The `training` extra is populated** (`torch`, `trl>=1.0`, `peft`, `transformers`, `vllm`,
   `bitsandbytes`, `accelerate`, `datasets`, `wandb`) but **never installed in CI** (`uv sync
   --frozen` installs default+dev only). `training/grpo_run.py` and `training/sft_warmup.py` import
   these lazily inside functions, so `mypy src` and CI import stay clean; mypy `ignore_missing_imports`
   overrides cover the modules actually imported.
4. **Record schema → v3.** Adding `SftDemo` (ADR-013) bumps the module `SCHEMA_VERSION` to 3, per the
   existing one-version-per-shape-change convention (ADR-009). Committed v1/v2 artifacts stay
   readable because every record self-describes its `schema_version`.

## Consequences

- The whole training wiring is validated on CPU/CI (`make grpo-dryrun` + the dry-run integration
  test) before any GPU spend.
- Regenerating `uv.lock` to add the `training` extra (required so CI's `--frozen` stays consistent)
  advanced numpy within its `>=2.0` range; the stricter numpy stubs surfaced one latent
  `no-any-return` in `ope/estimators.py`, fixed with a behavior-neutral `np.asarray` cast.
- The real GRPOTrainer step and vLLM path are exercised only on the user's GPU box; this session
  ships code + config + the CPU dry-run, with training metrics recorded as "TBD (pending run)".
