# 014 — `training-cpu` extra for a macOS/CPU tiny-real dry-run

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 3 (GRPO specialist training)

## Context

`uv sync --extra training` fails on macOS/arm64: the full extra pulls in `vllm` (and `bitsandbytes`),
which ship no macOS wheels (`vllm` → `nvidia-cudnn-frontend`, CUDA-only). But the tiny-real CPU
trainer dry-run — running a couple of real `GRPOTrainer` steps on a tiny model before renting a GPU —
needs only `torch`/`trl`/`peft`/`transformers`, none of which require CUDA. Developers must be able
to validate the real trainer wiring on a Mac.

## Decision

1. **Split the extra in two** (`pyproject.toml`):
   - **`training-cpu`** = `torch`, `trl`, `peft`, `transformers`, `accelerate`, `datasets`, `wandb`
     — no `vllm`, no `bitsandbytes`. Installs cleanly on macOS/CPU.
   - **`training`** = `specialist-router[training-cpu]` + `vllm` + `bitsandbytes` (the GPU superset).
   The self-reference keeps the two DRY.
2. **`make grpo-dryrun` runs the real tiny CPU trainer step** (`--real`, needs `training-cpu`):
   `grpo_run.dry_run_cpu` loads a tiny model + tokenizer on CPU (float32, **no 4-bit**, LoRA via
   `peft_config`), drives the env-coupled rollout through the sandboxed episode loop + verifier with
   a plain-`transformers` generation backend (**`use_vllm=False`**), and runs
   `GRPOTrainer.train()` for `dry_run.max_steps` steps. `make grpo-dryrun-mock` (and pytest) keep the
   dependency-free mock path for CI and quick checks.
3. **CI lock behavior is untouched.** CI still runs `uv sync --frozen` (default + dev only — no
   extras), so neither `training` nor `training-cpu` is installed there; `mypy`/`pytest` behave
   exactly as before. Regenerating `uv.lock` to add the new extra does not change what `--frozen`
   installs (verified: `torch` remains absent in the frozen env).

## Verified against the installed stack (TRL 1.8.0 on macOS/arm64)

Building `dry_run_cpu` against a real TRL install corrected the (previously speculative) rollout
wiring:

- TRL calls **`rollout_func(prompts, trainer)`** (positional trainer, not `**kwargs`); it must return
  `{prompt_ids, completion_ids, logprobs}` and reads a custom **`env_mask`** for the assistant/tool
  loss mask. `grpo_run.build_rollout_func` was fixed to this signature and maps each (group-expanded)
  prompt back to its task.
- **`max_prompt_length` is not a current `GRPOConfig` field** (as flagged in ADR-010) — confirmed
  absent; our config never passed it.
- `uv sync --extra training-cpu && make grpo-dryrun` runs 2 real `GRPOTrainer` steps on
  `trl-internal-testing/tiny-Qwen3ForCausalLM` on CPU in ~30 s, exercising
  rollout → verify → reward → optimizer step end-to-end. `uv run pytest -m training` (config-wiring
  check) also passes with the extra installed.

## Consequences

- Mac users validate the real trainer wiring before any GPU spend; the GPU box still uses the full
  `training` extra (vLLM fast generation + 4-bit QLoRA).
- With `training-cpu` installed locally, `mypy src` surfaces additional `no-untyped-call` /
  `attr-defined` findings from the real (largely untyped) torch/trl packages. These are **expected
  and out of scope**: per ADR-000/012 the extras are deliberately absent from the CI `mypy`
  environment, and adding `type: ignore` for them would fail `warn_unused_ignores` in the frozen CI
  env. `make lint` in CI (frozen) stays green.
