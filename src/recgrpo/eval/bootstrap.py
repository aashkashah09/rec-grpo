"""User-clustered bootstrap.

Evaluation examples from the same user are not independent -- one user
contributes many test interactions and the policy behaves similarly across all
of them -- so replicates resample users, not rows.
"""

from __future__ import annotations

import numpy as np


def user_clustered_bootstrap(
    values: np.ndarray,
    users: np.ndarray,
    n_replicates: int = 10000,
    seed: int = 42,
    statistic=np.mean,
) -> np.ndarray:
    """Resample users with replacement, recomputing `statistic` over their rows."""
    rng = np.random.default_rng(seed)
    unique_users, inverse = np.unique(users, return_inverse=True)
    by_user = [np.flatnonzero(inverse == u) for u in range(len(unique_users))]

    replicates = np.empty(n_replicates)
    for r in range(n_replicates):
        picks = rng.integers(0, len(unique_users), size=len(unique_users))
        rows = np.concatenate([by_user[p] for p in picks])
        replicates[r] = statistic(values[rows])
    return replicates


def bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    users: np.ndarray,
    n_replicates: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired difference b - a, with users resampled once per replicate.

    Both policies are scored on the same examples, so the replicate uses the
    same resampled users for each, which is what makes the interval a paired one.
    """
    rng = np.random.default_rng(seed)
    unique_users, inverse = np.unique(users, return_inverse=True)
    by_user = [np.flatnonzero(inverse == u) for u in range(len(unique_users))]

    replicates = np.empty(n_replicates)
    for r in range(n_replicates):
        picks = rng.integers(0, len(unique_users), size=len(unique_users))
        rows = np.concatenate([by_user[p] for p in picks])
        replicates[r] = values_b[rows].mean() - values_a[rows].mean()

    observed = float(values_b.mean() - values_a.mean())
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value: how often a replicate crosses zero
    tail = min((replicates <= 0).mean(), (replicates >= 0).mean())
    return {
        "observed": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(2 * tail),
        "n_replicates": n_replicates,
        "n_users": int(len(unique_users)),
    }
