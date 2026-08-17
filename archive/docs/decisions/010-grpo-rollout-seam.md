# 010 — Multi-turn env-coupled GRPO via TRL `rollout_func`

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 3 (GRPO specialist training)

## Context

Phase 3 trains Qwen3-4B-Instruct into a SQL specialist with TRL's GRPOTrainer. The environment is
**multi-turn tool use** (`inspect_schema`/`run_sql`/`python_calc` → `final_answer` over up to 12
turns), so a single prompt→completion GRPO cannot see SQL results. Current TRL (1.x) offers two
supported seams for multi-turn/agentic rollouts: `environment_factory` (TRL owns the loop and builds
the assistant/tool loss-mask automatically, but drives the model's **native** tool-calling) and
`rollout_func` (we own the generation loop and the token/mask/logprob plumbing, but can keep any
protocol). Overriding the private `_generate_and_score_completions` is no longer the right seam.

## Decision (confirmed with the user)

1. **Seam = `rollout_func`.** We keep the exact Phase-1/2 bespoke JSON action protocol,
   `ChatToolAgent`, the deterministic `verify()`, and the eval harness **byte-for-byte**. Phase 4
   becomes a pure checkpoint swap — the merged Phase-2 agents are untouched and train/serve/eval
   speak one protocol. The cost (owning the loss mask) is contained in `training/rollout.py` and
   unit-tested on CPU. `environment_factory` was the considered alternative; it gives automatic
   masking but forces a native-tool-calling migration of the merged Phase-2/4 serving+eval, so it
   was rejected for this project's reuse/parity priorities.
2. **Reuse all control flow.** `EnvRollout` drives `env.episode.run_episode` (which owns the 12-turn
   / tool budgets, dispatch, and the `Trajectory`); a `_RecordingAgent` records the token stream as
   a side effect. Budget/dispatch/verify logic keeps exactly one home.
3. **Generation and tokenization are injected** (`generate_fn`/`encode`) — vLLM + the model
   tokenizer in production, mocks in the dry-run. `training/rollout.py` never imports
   torch/trl/vLLM, so it (and the CPU dry-run) validate without a GPU.
4. **Reward cache keyed by a minted per-episode id.** For the `k`-th episode of a task in the batch
   the id is `f"{task_id}#{k}"`; the reward function reconstructs it from the batch's `task_id`
   column in order. It is deliberately **not** keyed by completion token ids: two episodes in one
   GRPO group can emit identical completions, and a content key would collide and cross-wire their
   rewards. `verify()` runs once per episode, in the rollout, and stays the single source of truth.
5. **GRPO signal.** Per episode `R_i = w_correct·1[correct] + w_format·format_score` (ADR-011); GRPO
   forms the group-relative advantage `(R_i − mean)/(std + ε)` and updates on assistant tokens only.
   The binary correctness term makes **group collapse** (all-G-correct or all-G-wrong ⇒ ~0 advantage)
   the primary failure mode; it is mitigated by the small nonzero format term, the pass-rate
   curriculum in `TaskSampler`, and a large-enough `num_generations`.

## Scope note (CONVENTIONS.md rule #2 / PROJECT_PLAN §1.2)

GRPO's internal clipped token-level ratio is a **training objective**, not off-policy evaluation.
There is no trajectory-level importance sampling of the agent anywhere: the specialist is evaluated
only by held-out task success (`evaluation/harness.py`). OPE remains confined to the single routing
decision (Phase 2). This distinction is stated so a reviewer never conflates the two.

## Consequences

- Phase 4 is a checkpoint swap behind the existing `Agent`/`ArmRunner` interfaces.
- `training/grpo_run.py` is the only module that touches TRL/torch/vLLM, all via lazy imports, and
  is never run in CI. The exact TRL 1.x parameter names are mirrored in `configs/grpo.yaml` and
  re-verified against the pinned TRL by `tests/integration/test_grpo_trl_wiring.py` (marked
  `training`); TRL's agentic surface moves fast, so pin versions on the GPU box and re-verify.
- The masking/token/logprob plumbing is the main risk of this seam; it is isolated in `EnvRollout`
  and covered by the CPU dry-run's contract checks.
