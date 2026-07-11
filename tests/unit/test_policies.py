"""Unit tests for the bandit routing policies."""

from __future__ import annotations

import numpy as np
import pytest

from specialist_router.config import (
    EpsilonGreedyConfig,
    LinUCBConfig,
    RewardConfig,
    RouterConfig,
    ThompsonConfig,
)
from specialist_router.env.records import ARMS
from specialist_router.router.policies import (
    ConstantPolicy,
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    ThompsonLogisticPolicy,
    UniformPolicy,
    build_target_policies,
)
from tests.support import make_logged_dataset

REWARD = RewardConfig(lambda_cost=0.3, mu_latency=0.1, cost_ref_usd=0.02, latency_ref_s=8.0)


def _probs_sum_to_one(props: dict[str, float]) -> None:
    assert set(props) == set(ARMS)
    assert sum(props.values()) == pytest.approx(1.0)


def test_uniform_is_half_half() -> None:
    pol = UniformPolicy()
    props = pol.propensities(np.array([1.0, 0.2]))
    assert props == {"local": 0.5, "api": 0.5}
    _probs_sum_to_one(props)


def test_uniform_select_returns_logged_propensity() -> None:
    pol = UniformPolicy()
    rng = np.random.default_rng(0)
    arm, prop = pol.select(np.array([1.0, 0.0]), rng)
    assert arm in ARMS
    assert prop == 0.5


def test_epsilon_greedy_exact_propensities() -> None:
    data = make_logged_dataset(200, 4, seed=1)
    pol = EpsilonGreedyPolicy(EpsilonGreedyConfig(epsilon=0.1, ridge_lambda=1.0)).fit(data)
    props = pol.propensities(data.features[0])
    _probs_sum_to_one(props)
    # One arm is greedy (1 - eps + eps/2), the other is the floor (eps/2).
    assert max(props.values()) == pytest.approx(0.95)
    assert min(props.values()) == pytest.approx(0.05)


def test_linucb_full_support_and_sums_to_one() -> None:
    data = make_logged_dataset(200, 4, seed=2)
    pol = LinUCBPolicy(LinUCBConfig(alpha=1.0, ridge_lambda=1.0, temperature=0.25)).fit(data)
    props = pol.propensities(data.features[3])
    _probs_sum_to_one(props)
    assert all(p > 0.0 for p in props.values())


def test_thompson_positive_and_deterministic() -> None:
    data = make_logged_dataset(200, 4, seed=3)
    pol = ThompsonLogisticPolicy(
        ThompsonConfig(prior_variance=1.0, newton_steps=25, n_posterior_samples=128), REWARD
    ).fit(data)
    x = data.features[5]
    first = pol.propensities(x)
    second = pol.propensities(x)
    _probs_sum_to_one(first)
    assert all(p > 0.0 for p in first.values())
    assert first == second  # propensities are a deterministic function of the context


def test_constant_policy_puts_all_mass_on_one_arm() -> None:
    pol = ConstantPolicy("api")
    props = pol.propensities(np.array([1.0, 0.0]))
    assert props == {"local": 0.0, "api": 1.0}
    arm, prop = pol.select(np.array([1.0, 0.0]), np.random.default_rng(0))
    assert (arm, prop) == ("api", 1.0)


def test_unfit_policy_raises() -> None:
    pol = EpsilonGreedyPolicy(EpsilonGreedyConfig(epsilon=0.1, ridge_lambda=1.0))
    with pytest.raises(RuntimeError, match="fit must be called"):
        pol.propensities(np.array([1.0, 0.0, 0.0, 0.0]))


def test_build_target_policies_returns_all(router_config: RouterConfig) -> None:
    data = make_logged_dataset(200, router_config.features.embed_dim + 13, seed=4)
    policies = build_target_policies(router_config, data)
    assert set(policies) == {
        "uniform",
        "epsilon_greedy",
        "linucb",
        "thompson_logistic",
        "always_local",
        "always_api",
    }
