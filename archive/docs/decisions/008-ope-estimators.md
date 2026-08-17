# 008 — OPE estimators, DR cross-fitting, and the breakage study

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Router + serving + logging + OPE)

## Context

Phase 2 evaluates candidate routing policies off-policy from Uniform-logged data (IPS/SNIPS/DR +
direct method, bootstrap CIs, ESS), validated by replay-A/B, with a "when estimators break"
mini-study (`PROJECT_PLAN.md` §3). `CONVENTIONS.md` rule #2 hard-limits the scope.

## Decisions

1. **Scope: the single routing decision only.** Every estimator is a function of a
   `LoggedDataset` row — one context, one arm, one scalar reward. There is **no** trajectory-level
   importance sampling of the agent's tool-use rollout, and the module docstring states this. The
   specialist is evaluated separately by held-out verifier success (Phase 3+).
2. **Per-row contributions drive everything.** `estimators.py` exposes each estimator as a function
   of precomputed per-row arrays (`ips_terms`, `dm_terms`, `dr_terms`, `weights`), so `ci.py`
   bootstraps CIs by a single vectorised index-resample without duplicating estimator logic.
3. **DR uses a cross-fitted outcome model.** `q̂(x, a)` is fit out-of-fold (K folds, ridge by
   default; optional GBM behind the `ml` extra), so the DR correction is never evaluated on rows the
   model trained on — avoiding optimistic bias. A `q_override` seam lets the property tests inject a
   deliberately misspecified `q̂` and show DR still tracks truth while the direct method does not.
4. **The breakage study uses the tabular simulator, not the router logs.** Because
   `BanditSimulator` has a closed-form true value, the study measures each estimator's **bias and
   variance against ground truth** as logging overlap shrinks — a rigorous demonstration (IPS
   variance explodes, SNIPS/DR steadier, ESS collapses) rather than an unfalsifiable plot.
5. **Overlap is guaranteed by design.** The logging policy is Uniform (propensity 0.5), so weights
   are bounded (≤ 2) and every OPE table reports ESS; deterministic target policies (always-local/
   always-api) are valid targets (rows the target wouldn't take get weight 0).

## Consequences

- The estimators are pure and property-tested against known truth (IPS unbiased in expectation,
  SNIPS/DR ≤ IPS variance, DR robust to a misspecified model).
- The GBM outcome model and any non-ridge path are optional and not exercised in CI (kept torch-
  and sklearn-free by default); the coverage gap there is intentional and honest.
