# 007 — Reward definition, λ/μ defaults, normalization, and no LLM-judge in Phase 2

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Router + serving + logging + OPE)

## Context

The routing reward is `quality − λ·cost − μ·latency` (`PROJECT_PLAN.md` §3). `CONVENTIONS.md` says the
reward weights are decided with the user, not guessed, and the plan calls for a "calibrated
LLM-judge fallback where verifier N/A." Both points were resolved with the user before coding.

## Decisions

1. **Weights (user-approved): λ = 0.3, μ = 0.1.** With normalized cost/latency in `[0, 1]`, λ and μ
   are exactly the worst-case penalties in quality units.
2. **Fixed reference-scale normalization (user-approved), not dataset min/max.**
   `cost_norm = min(cost_usd / cost_ref_usd, 1)`, `latency_norm = min(latency_s / latency_ref_s, 1)`
   with `cost_ref_usd = 0.02`, `latency_ref_s = 8.0` in `configs/router.yaml`. Fixed scales keep the
   reward **stationary** across logs and prevent leakage across the OPE train/replay split;
   dataset min/max would make the reward depend on the sample and couple the two splits.
3. **Sensitivity sweep.** Because the reward is a post-hoc function of logged components, the λ/μ
   sweep re-scores the *same* decisions with no new traffic (planned grid
   λ ∈ {0, .1, .2, .3, .5, 1}, μ ∈ {0, .05, .1, .2}); the reward weights are logged on every
   `RouterDecision` so any re-scoring is reproducible.
4. **No LLM-judge in Phase 2 (user-approved deviation from the plan wording).** All eight templates
   have deterministic verifiers, so `quality` is always the verifier verdict. Implementing an
   unused judge would add complexity that outruns the artifacts (`CONVENTIONS.md`). We instead document
   the seam: quality would flow through a `quality_source: verifier | judge` switch; a judge, if
   ever needed for a free-form template, would be **calibrated against verifier labels on a held-out
   split (reporting agreement + Cohen's κ) and gated behind `@pytest.mark.external`**. It is left
   unimplemented by design.

## Consequences

- The reward is a small pure function (`router/reward.py`), unit-tested to 100%, and every logged
  decision is independently re-scorable.
- Replay-A/B asserts the logged and replayed decisions share one `(λ, μ, cost_ref, latency_ref)`
  signature (`ope/replay.py`), so predicted and realized values are always on the same scale.
- With these particular stub cost/latency scales the reward-optimal policy is close to
  always-frontier; the *frontier* (cost vs quality) is where routing visibly pays off, and the λ/μ
  sweep is what surfaces the regime where the learned router wins on reward too. Reported honestly
  in the README.
