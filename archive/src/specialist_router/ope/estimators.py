"""Off-policy estimators for the single routing decision: IPS, SNIPS, the direct method, and DR.

**Scope (``CONVENTIONS.md`` rule #2 / ``PROJECT_PLAN`` §1.2):** every estimator here evaluates the
*routing decision alone* — one context, one arm, one scalar reward. There is no trajectory-level
importance sampling of the agent's tool-use rollout, and none must ever be added. The unit of
analysis is a :class:`~specialist_router.router.logger.LoggedDataset` row.

Given a target policy ``π_e`` and a Uniform-logged dataset with propensities ``π₀``:

* **IPS** — ``mean( w_i · r_i )`` with ``w_i = π_e(a_i|x_i) / π₀(a_i|x_i)``. Unbiased when
  ``π₀`` has support wherever ``π_e`` does (guaranteed here — logging is Uniform).
* **SNIPS** — ``Σ w_i r_i / Σ w_i``. Self-normalized: biased but far lower variance than IPS.
* **Direct method (DM)** — ``mean( Σ_a π_e(a|x_i) q̂(x_i, a) )`` from a fitted outcome model ``q̂``.
  Low variance, but biased if ``q̂`` is misspecified.
* **Doubly robust (DR)** — DM plus an IPS correction on the residuals: consistent if *either*
  ``q̂`` or the propensities are right. ``q̂`` is **cross-fitted** (out-of-fold) so the correction
  is not evaluated on rows the model trained on.

The per-row contributions are exposed so :mod:`specialist_router.ope.ci` can bootstrap CIs without
re-deriving any estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist_router.config import OpeConfig
from specialist_router.env.records import ARMS
from specialist_router.router.logger import LoggedDataset
from specialist_router.router.policies import Policy, _ridge_fit

_N_ARMS = len(ARMS)


def policy_action_matrix(data: LoggedDataset, policy: Policy) -> npt.NDArray[np.float64]:
    """Return the ``(n, n_arms)`` matrix of target propensities ``π_e(a | x_i)`` for every arm."""
    matrix = np.empty((data.n, _N_ARMS), dtype=np.float64)
    for i in range(data.n):
        propensities = policy.propensities(data.features[i])
        matrix[i] = [propensities[a] for a in ARMS]
    return matrix


def importance_weights(
    data: LoggedDataset, action_matrix: npt.NDArray[np.float64], weight_clip: float | None
) -> npt.NDArray[np.float64]:
    """Per-row weights ``π_e(a_i|x_i) / π₀(a_i|x_i)``, optionally clipped at ``weight_clip``."""
    target_prop = action_matrix[np.arange(data.n), data.action_index]
    weights = target_prop / data.propensity
    if weight_clip is not None:
        weights = np.minimum(weights, weight_clip)
    # Explicit cast: newer numpy (>=2.3) stubs type ``__truediv__``/``minimum`` as ``Any``, so pin
    # the declared float64 array return (behavior-neutral; ``weights`` is already float64).
    return np.asarray(weights, dtype=np.float64)


def cross_fitted_q(
    data: LoggedDataset, ridge_lambda: float, n_folds: int, seed: int
) -> npt.NDArray[np.float64]:
    """Cross-fitted per-arm outcome model ``q̂(x_i, a)`` for every arm (shape ``(n, n_arms)``).

    Rows are partitioned into ``n_folds`` folds; each fold's predictions come from a per-arm ridge
    fit on the *other* folds, so no row's ``q̂`` is contaminated by its own reward. Arms with no
    training rows in a fold predict 0 (their DR correction still carries the signal).
    """
    folds = min(n_folds, data.n)
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, folds, size=data.n)
    q = np.zeros((data.n, _N_ARMS), dtype=np.float64)
    for f in range(folds):
        test = assignment == f
        train = ~test
        if not bool(test.any()):
            continue
        for ai, arm in enumerate(ARMS):
            arm_train = train & data.arm_mask(arm)
            if int(arm_train.sum()) >= 1:
                weights = _ridge_fit(data.features[arm_train], data.reward[arm_train], ridge_lambda)
                q[test, ai] = data.features[test] @ weights
    return q


@dataclass(frozen=True, slots=True)
class OpeContributions:
    """Per-row terms each estimator is a function of (so bootstrap resampling is trivial)."""

    weights: npt.NDArray[np.float64]
    """Importance weights ``w_i`` (already clipped if configured)."""

    ips_terms: npt.NDArray[np.float64]
    """``w_i · r_i``."""

    dm_terms: npt.NDArray[np.float64]
    """``Σ_a π_e(a|x_i) q̂(x_i, a)`` — the direct-method per-row value."""

    dr_terms: npt.NDArray[np.float64]
    """``dm_i + w_i (r_i − q̂(x_i, a_i))`` — the doubly-robust per-row value."""


def compute_contributions(
    data: LoggedDataset,
    policy: Policy,
    config: OpeConfig,
    q_override: npt.NDArray[np.float64] | None = None,
) -> OpeContributions:
    """Compute the per-row estimator contributions for ``policy`` on ``data``.

    Args:
        data: The logged dataset.
        policy: The target policy to evaluate.
        config: OPE settings (weight clip, DR outcome model).
        q_override: An optional ``(n, n_arms)`` outcome model to use *instead of* fitting one.
            Used by the property tests to inject a deliberately misspecified ``q̂`` and show DR
            stays consistent (the propensities are correct) while the direct method does not.
    """
    action_matrix = policy_action_matrix(data, policy)
    weights = importance_weights(data, action_matrix, config.weight_clip)
    q = _fit_outcome_model(data, config) if q_override is None else q_override
    q_logged = q[np.arange(data.n), data.action_index]
    dm_terms = np.sum(action_matrix * q, axis=1)
    dr_terms = dm_terms + weights * (data.reward - q_logged)
    return OpeContributions(
        weights=weights,
        ips_terms=weights * data.reward,
        dm_terms=dm_terms,
        dr_terms=dr_terms,
    )


def _fit_outcome_model(data: LoggedDataset, config: OpeConfig) -> npt.NDArray[np.float64]:
    """Fit the cross-fitted outcome model (ridge, or GBM via the optional ``ml`` extra)."""
    if config.dr.reward_model == "gbm":
        return _cross_fitted_gbm(data, config.dr.n_folds, config.seed)
    return cross_fitted_q(data, config.dr.ridge_lambda, config.dr.n_folds, config.seed)


def _cross_fitted_gbm(data: LoggedDataset, n_folds: int, seed: int) -> npt.NDArray[np.float64]:
    """Cross-fitted per-arm gradient-boosted outcome model (optional; needs scikit-learn)."""
    from sklearn.ensemble import GradientBoostingRegressor  # lazy: optional 'ml' extra

    folds = min(n_folds, data.n)
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, folds, size=data.n)
    q = np.zeros((data.n, _N_ARMS), dtype=np.float64)
    for f in range(folds):
        test = assignment == f
        train = ~test
        if not bool(test.any()):
            continue
        for ai, arm in enumerate(ARMS):
            arm_train = train & data.arm_mask(arm)
            if int(arm_train.sum()) >= 2:
                model = GradientBoostingRegressor(random_state=seed)
                model.fit(data.features[arm_train], data.reward[arm_train])
                q[test, ai] = model.predict(data.features[test])
    return q


def ips(contributions: OpeContributions) -> float:
    """The IPS point estimate ``mean(w_i r_i)``."""
    return float(np.mean(contributions.ips_terms))


def snips(contributions: OpeContributions) -> float:
    """The SNIPS point estimate ``Σ w_i r_i / Σ w_i`` (falls back to 0 if all weights vanish)."""
    total_weight = float(np.sum(contributions.weights))
    if total_weight == 0.0:
        return 0.0
    return float(np.sum(contributions.ips_terms) / total_weight)


def direct_method(contributions: OpeContributions) -> float:
    """The direct-method point estimate ``mean(Σ_a π_e(a|x) q̂(x, a))``."""
    return float(np.mean(contributions.dm_terms))


def doubly_robust(contributions: OpeContributions) -> float:
    """The doubly-robust point estimate."""
    return float(np.mean(contributions.dr_terms))


def effective_sample_size(weights: npt.NDArray[np.float64]) -> float:
    """Kish effective sample size ``(Σ w)² / Σ w²`` (0 if all weights are 0)."""
    denom = float(np.sum(weights**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / denom)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Point estimates and importance-weight diagnostics for one target policy."""

    policy_name: str
    n: int
    ips: float
    snips: float
    direct_method: float
    doubly_robust: float
    ess: float
    ess_fraction: float
    max_weight: float
    mean_weight: float
    contributions: OpeContributions


def evaluate_policy(
    data: LoggedDataset,
    policy: Policy,
    config: OpeConfig,
    q_override: npt.NDArray[np.float64] | None = None,
) -> PolicyEvaluation:
    """Evaluate ``policy`` off-policy on ``data`` with all four estimators plus ESS diagnostics."""
    contributions = compute_contributions(data, policy, config, q_override)
    ess = effective_sample_size(contributions.weights)
    return PolicyEvaluation(
        policy_name=policy.name,
        n=data.n,
        ips=ips(contributions),
        snips=snips(contributions),
        direct_method=direct_method(contributions),
        doubly_robust=doubly_robust(contributions),
        ess=ess,
        ess_fraction=ess / data.n if data.n else 0.0,
        max_weight=float(np.max(contributions.weights)) if data.n else 0.0,
        mean_weight=float(np.mean(contributions.weights)) if data.n else 0.0,
        contributions=contributions,
    )
