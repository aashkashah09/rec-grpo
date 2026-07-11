# 013 — Optional SFT warmup: trigger, demo generation, and spend safety

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 3 (GRPO specialist training)

## Context

PROJECT_PLAN §3 allows an optional SFT warmup (500–1500 tool-format demos) if the base model's
tool-format compliance is below ~60%, generated via the frontier API. It must be kept clearly
separate from the RL run, and `CONVENTIONS.md` requires care around anything that spends money.

## Decision

1. **Trigger.** `probe_compliance` measures base-model tool-format compliance on a small probe set
   (the eval harness's format-rate). Warmup is indicated only if compliance `<
   sft.trigger_compliance_threshold` (default 0.60). The probe uses the local (base-model) endpoint,
   so it incurs no frontier spend.
2. **Demos.** `generate_demos` runs the Phase-2 frontier `api_agent` over training-split tasks and
   keeps **only verifier-correct episodes** (so the warmup teaches *correct* tool use, not merely
   well-formed tool use), serialized as versioned `SftDemo` JSONL (schema v3). It reuses
   `configs/serving.yaml → api_endpoint`, so there is **no new provider choice** — provider stays
   pluggable (ADR-005).
3. **Spend safety (user rider).** `scripts/sft_warmup.py` **prints an estimated API cost and makes no
   frontier call unless `--confirm-spend` is passed**; without it, it prints the estimate and exits
   with code 2. The estimate is an a-priori projection from fixed token assumptions and the
   endpoint's prices, explicitly labeled a *lower bound* (a frontier pass rate < 1 needs more
   attempts per kept demo). No cost is ever presented as a measured result.
4. **Separation from RL.** The warmup has its own entrypoint, its own config block (`grpo.yaml →
   sft`), its own output dir, and its own W&B run. GRPO consumes a warmup adapter **only** via an
   explicit `grpo_run --init-from-sft <adapter>`; it is never merged automatically.

## Consequences

- The RL run never implicitly triggers SFT or spends API budget; both are opt-in and gated.
- `estimate_demo_cost` and the `--confirm-spend` gate are unit-tested (`tests/unit/test_sft_warmup`);
  the gate test asserts no API path is reached without confirmation.
- The `SFTTrainer` step itself is GPU-only (lazy imports) and not run in this session.
