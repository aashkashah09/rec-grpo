# 011 — GRPO training reward: verifier correctness + small format term

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 3 (GRPO specialist training)

## Context

`CLAUDE.md` requires the training reward to be the Phase-1 deterministic verifier plus a small
format/tool-compliance reward, and that reward weights be decided with the user, not guessed
(cf. ADR-007 for the *routing* reward). The training reward is a distinct quantity from the routing
reward (`router/reward.py`, `quality − λ·cost − μ·latency`) and lives in `training/reward.py`.

## Decision (weights confirmed with the user)

1. **`R = w_correct·1[correct] + w_format·format_score`** with **`w_correct = 1.0`, `w_format =
   0.15`** (additive, small). Correctness is `Verdict.correct` from the same pure verifier used
   everywhere; there is no LLM judge.
2. **`format_score ∈ [0, 1]`** is the mean of four components, each in `[0, 1]`:
   `all_actions_parse` (fraction of the model's generations that parsed to a valid action),
   `used_tool_before_answer`, `well_formed_final_answer` (ended by submitting a correctly-typed,
   non-fallback answer within budget), and `within_budget`. The rollout measures `all_actions_parse`
   and the final-answer flag precisely (it sees every raw generation); the eval harness derives
   proxies from the `Trajectory` alone.
3. **Why additive-small (not a hard gate or a larger weight):** the small term keeps a nonzero
   gradient when a whole GRPO group is all-correct or all-wrong (binary reward → ~0 advantage) and
   breaks ties, but is deliberately too small to prefer a well-formatted wrong answer over a correct
   one (`0.15 < 1.0`), so it cannot be reward-hacked into beating correctness. This matches the
   plan's expected trajectory: format saturates first, accuracy climbs after.

## Consequences

- The reward is a small pure function, unit-tested near-completely (`tests/unit/test_training_reward`).
- **Format-rate is logged next to success on the held-out set as a drift guard:** a falling
  format-rate while reward/success rises is a warning of protocol drift or verifier gaming
  (`FormatRateTrend.drift_warning`, `evaluation/harness.py`). Because `format_score` is independent
  of correctness, the two curves can move independently and the divergence is informative.
- If a reward-hacking case study is later observed (PROJECT_PLAN bonus), the verifier — not the
  format term — is the thing to harden, since correctness can only come from deterministic ground
  truth.
