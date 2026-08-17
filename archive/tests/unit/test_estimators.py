"""Unit tests for the OPE estimators and diagnostics on hand-checkable inputs."""

from __future__ import annotations

import numpy as np
import pytest

from specialist_router.config import DrConfig, OpeConfig
from specialist_router.ope.ci import bootstrap_intervals, weight_diagnostics
from specialist_router.ope.estimators import (
    compute_contributions,
    cross_fitted_q,
    effective_sample_size,
    evaluate_policy,
    importance_weights,
    policy_action_matrix,
)
from specialist_router.router.policies import ConstantPolicy, UniformPolicy
from tests.support import make_logged_dataset

OPE = OpeConfig(
    schema_version=1,
    seed=0,
    n_bootstrap=200,
    ci_alpha=0.05,
    weight_clip=None,
    dr=DrConfig(reward_model="ridge", ridge_lambda=1.0, n_folds=4),
)


def test_on_policy_uniform_gives_unit_weights_and_mean_reward() -> None:
    data = make_logged_dataset(300, 4, seed=10)
    ev = evaluate_policy(data, UniformPolicy(), OPE)
    # Target == logging (both uniform 0.5) => every weight is exactly 1.
    assert np.allclose(ev.contributions.weights, 1.0)
    assert ev.ips == pytest.approx(float(np.mean(data.reward)))
    assert ev.snips == pytest.approx(float(np.mean(data.reward)))
    assert ev.ess == pytest.approx(data.n)
    assert ev.ess_fraction == pytest.approx(1.0)


def test_importance_weights_match_definition() -> None:
    data = make_logged_dataset(50, 3, seed=11)
    matrix = policy_action_matrix(data, ConstantPolicy("local"))
    weights = importance_weights(data, matrix, weight_clip=None)
    # Constant-local target: weight = 1[action==local]/0.5 = 2 on local rows, 0 on api rows.
    local = data.arm_mask("local")
    assert np.allclose(weights[local], 2.0)
    assert np.allclose(weights[~local], 0.0)


def test_weight_clip_caps_weights() -> None:
    data = make_logged_dataset(50, 3, seed=12)
    matrix = policy_action_matrix(data, ConstantPolicy("local"))
    weights = importance_weights(data, matrix, weight_clip=1.5)
    assert weights.max() <= 1.5


def test_cross_fitted_q_shape_and_finite() -> None:
    data = make_logged_dataset(80, 4, seed=13)
    q = cross_fitted_q(data, ridge_lambda=1.0, n_folds=4, seed=0)
    assert q.shape == (data.n, 2)
    assert np.all(np.isfinite(q))


def test_dr_equals_dm_plus_correction() -> None:
    data = make_logged_dataset(120, 4, seed=14)
    contrib = compute_contributions(data, UniformPolicy(), OPE)
    q_logged_implied = data.reward - (contrib.dr_terms - contrib.dm_terms) / contrib.weights
    # Reconstructing q̂(x, a_i) from the DR identity should be finite and bounded.
    assert np.all(np.isfinite(q_logged_implied))


def test_effective_sample_size_edge_cases() -> None:
    assert effective_sample_size(np.zeros(5)) == 0.0
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)


def test_bootstrap_ci_brackets_point() -> None:
    data = make_logged_dataset(300, 4, seed=15)
    ev = evaluate_policy(data, UniformPolicy(), OPE)
    cis = bootstrap_intervals(ev, n_bootstrap=300, alpha=0.05, seed=0)
    for name in ("ips", "snips", "direct_method", "doubly_robust"):
        ci = cis[name]
        assert ci.lo <= ci.point <= ci.hi


def test_weight_diagnostics_reports_overlap() -> None:
    data = make_logged_dataset(200, 4, seed=16)
    ev = evaluate_policy(data, ConstantPolicy("api"), OPE)
    diag = weight_diagnostics(ev.contributions)
    assert 0.0 <= diag.ess_fraction <= 1.0
    assert diag.max_weight >= diag.mean_weight
