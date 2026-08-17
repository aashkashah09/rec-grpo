"""Shared test builders for router decisions and logged datasets."""

from __future__ import annotations

import numpy as np

from specialist_router.env.records import Arm, RouterDecision
from specialist_router.router.logger import LoggedDataset


def make_decision(
    *,
    decision_id: str = "dec-0",
    action: Arm = "local",
    propensity: float = 0.5,
    reward: float = 0.5,
    quality: int = 1,
    reward_lambda: float = 0.3,
    reward_mu: float = 0.1,
    cost_ref_usd: float = 0.02,
    latency_ref_s: float = 8.0,
    feature_vector: list[float] | None = None,
) -> RouterDecision:
    """Build a valid :class:`RouterDecision` with sensible defaults, overridable per field."""
    vec = feature_vector if feature_vector is not None else [1.0, 0.5]
    return RouterDecision(
        decision_id=decision_id,
        task_id="task-0",
        template_id="revenue_by_segment",
        difficulty="easy",
        feature_names=[f"f{i}" for i in range(len(vec))],
        feature_vector=vec,
        feature_dim=len(vec),
        policy_name="uniform",
        policy_version="1",
        policy_params_hash="uniform",
        action=action,
        propensity=propensity,
        all_propensities={"local": 0.5, "api": 0.5},
        quality=quality,
        cost_usd=0.001,
        latency_s=0.5,
        cost_norm=0.05,
        latency_norm=0.06,
        reward=reward,
        reward_lambda=reward_lambda,
        reward_mu=reward_mu,
        cost_ref_usd=cost_ref_usd,
        latency_ref_s=latency_ref_s,
        agent_versions={"local": "stub-local", "api": "stub-api"},
        seed=0,
        timestamp="2026-07-11T00:00:00+000d",
    )


def make_logged_dataset(n: int, d: int, seed: int) -> LoggedDataset:
    """Build a small two-arm :class:`LoggedDataset` with Uniform (0.5) logging propensities."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, d))
    features[:, 0] = 1.0  # bias column
    action_index = rng.integers(0, 2, size=n).astype(np.intp)
    reward = rng.random(n)
    quality = (reward > 0.5).astype(np.intp)
    return LoggedDataset(
        features=features,
        action_index=action_index,
        propensity=np.full(n, 0.5),
        reward=reward,
        quality=quality,
        cost_norm=rng.random(n) * 0.1,
        latency_norm=rng.random(n) * 0.1,
    )
