"""Bootstrap confidence intervals and effective-sample-size diagnostics for the OPE estimators.

CIs are percentile bootstrap over the *decisions*: resample logged rows with replacement, recompute
each estimator from its precomputed per-row contributions, and take empirical quantiles. Because
:mod:`specialist_router.ope.estimators` exposes every estimator as a function of per-row arrays,
the bootstrap is a single vectorised index-resample — no estimator logic is duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist_router.ope.estimators import (
    OpeContributions,
    PolicyEvaluation,
    effective_sample_size,
)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate with a percentile-bootstrap confidence interval."""

    point: float
    lo: float
    hi: float


def _percentile_ci(
    point: float, samples: npt.NDArray[np.float64], alpha: float
) -> ConfidenceInterval:
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return ConfidenceInterval(point=point, lo=lo, hi=hi)


def bootstrap_intervals(
    evaluation: PolicyEvaluation, n_bootstrap: int, alpha: float, seed: int
) -> dict[str, ConfidenceInterval]:
    """Bootstrap ``(1 − alpha)`` CIs for IPS, SNIPS, the direct method, and DR.

    Args:
        evaluation: A policy evaluation carrying the per-row contributions.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Two-sided miscoverage (0.05 → 95% CIs).
        seed: RNG seed for reproducible resampling.

    Returns:
        A mapping estimator-name → :class:`ConfidenceInterval`.
    """
    contributions = evaluation.contributions
    n = evaluation.n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    ips_samples = contributions.ips_terms[idx].mean(axis=1)
    dm_samples = contributions.dm_terms[idx].mean(axis=1)
    dr_samples = contributions.dr_terms[idx].mean(axis=1)
    weight_sums = contributions.weights[idx].sum(axis=1)
    ips_sums = contributions.ips_terms[idx].sum(axis=1)
    safe_sums = np.where(weight_sums > 0.0, weight_sums, 1.0)
    snips_samples = np.where(weight_sums > 0.0, ips_sums / safe_sums, 0.0)

    return {
        "ips": _percentile_ci(evaluation.ips, ips_samples, alpha),
        "snips": _percentile_ci(evaluation.snips, snips_samples, alpha),
        "direct_method": _percentile_ci(evaluation.direct_method, dm_samples, alpha),
        "doubly_robust": _percentile_ci(evaluation.doubly_robust, dr_samples, alpha),
    }


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    """Importance-weight health for an evaluation (the overlap/variance warning signs)."""

    ess: float
    ess_fraction: float
    max_weight: float
    mean_weight: float


def weight_diagnostics(contributions: OpeContributions) -> WeightDiagnostics:
    """Summarise importance-weight overlap: ESS, ESS fraction, and max/mean weight."""
    weights = contributions.weights
    n = int(weights.shape[0])
    ess = effective_sample_size(weights)
    return WeightDiagnostics(
        ess=ess,
        ess_fraction=ess / n if n else 0.0,
        max_weight=float(np.max(weights)) if n else 0.0,
        mean_weight=float(np.mean(weights)) if n else 0.0,
    )
