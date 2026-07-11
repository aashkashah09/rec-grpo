"""Property tests: OPE estimators vs a simulator with an analytically known policy value.

These assert the statistical guarantees the estimators are supposed to have — IPS unbiased in
expectation, SNIPS/DR lower variance than IPS, and DR's robustness to a misspecified outcome
model — against ground truth from :class:`BanditSimulator`.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from specialist_router.config import DrConfig, OpeConfig
from specialist_router.ope.estimators import compute_contributions, evaluate_policy, ips
from specialist_router.ope.simulator import BanditSimulator, TabularPolicy

OPE = OpeConfig(
    schema_version=1,
    seed=0,
    n_bootstrap=100,
    ci_alpha=0.05,
    weight_clip=None,
    dr=DrConfig(reward_model="ridge", ridge_lambda=1.0, n_folds=4),
)

REWARD_MEANS = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.55]], dtype=np.float64)
CONTEXT_PROBS = np.array([0.34, 0.33, 0.33], dtype=np.float64)


def _uniform_sim() -> BanditSimulator:
    logging = np.array([[0.5, 0.5]] * 3, dtype=np.float64)
    return BanditSimulator(CONTEXT_PROBS, REWARD_MEANS, logging)


def _target(p0: float, p1: float, p2: float) -> tuple[TabularPolicy, np.ndarray]:
    table = np.array([[p0, 1 - p0], [p1, 1 - p1], [p2, 1 - p2]], dtype=np.float64)
    return TabularPolicy(table, name="target"), table


@settings(max_examples=12, deadline=None)
@given(
    st.floats(0.2, 0.8),
    st.floats(0.2, 0.8),
    st.floats(0.2, 0.8),
)
def test_ips_unbiased_in_expectation(p0: float, p1: float, p2: float) -> None:
    sim = _uniform_sim()
    target, table = _target(p0, p1, p2)
    truth = sim.true_value(table)
    values = [evaluate_policy(sim.sample(1500, seed=1000 + s), target, OPE).ips for s in range(30)]
    assert abs(float(np.mean(values)) - truth) < 0.02


@settings(max_examples=12, deadline=None)
@given(
    st.floats(0.2, 0.8),
    st.floats(0.2, 0.8),
    st.floats(0.2, 0.8),
)
def test_dr_unbiased_in_expectation(p0: float, p1: float, p2: float) -> None:
    sim = _uniform_sim()
    target, table = _target(p0, p1, p2)
    truth = sim.true_value(table)
    values = [
        evaluate_policy(sim.sample(1500, seed=2000 + s), target, OPE).doubly_robust
        for s in range(30)
    ]
    assert abs(float(np.mean(values)) - truth) < 0.02


def test_snips_and_dr_have_no_more_variance_than_ips() -> None:
    # Skewed logging so importance weights vary — the regime where SNIPS/DR help most.
    logging = np.array([[0.25, 0.75]] * 3, dtype=np.float64)
    sim = BanditSimulator(CONTEXT_PROBS, REWARD_MEANS, logging)
    target, _ = _target(0.8, 0.2, 0.5)
    ips_vals, snips_vals, dr_vals = [], [], []
    for s in range(120):
        ev = evaluate_policy(sim.sample(1500, seed=3000 + s), target, OPE)
        ips_vals.append(ev.ips)
        snips_vals.append(ev.snips)
        dr_vals.append(ev.doubly_robust)
    ips_std = float(np.std(ips_vals))
    assert float(np.std(snips_vals)) <= ips_std + 1e-3
    assert float(np.std(dr_vals)) <= ips_std + 1e-3


def test_dr_robust_to_misspecified_outcome_model() -> None:
    # With a deliberately useless q̂ = 0, DR falls back to IPS (unbiased), while the direct
    # method inherits the model's bias. Propensities are correct, so DR still tracks truth.
    sim = _uniform_sim()
    target, table = _target(0.8, 0.2, 0.5)
    truth = sim.true_value(table)
    dm_vals, dr_vals = [], []
    for s in range(30):
        data = sim.sample(1500, seed=4000 + s)
        bad_q = np.zeros((data.n, 2), dtype=np.float64)
        contrib = compute_contributions(data, target, OPE, q_override=bad_q)
        dm_vals.append(float(np.mean(contrib.dm_terms)))
        dr_vals.append(float(np.mean(contrib.dr_terms)))
        # DR with q=0 is exactly IPS.
        assert float(np.mean(contrib.dr_terms)) == ips(contrib)
    assert abs(float(np.mean(dm_vals))) < 0.05  # direct method collapses to ~0 (biased)
    assert abs(float(np.mean(dr_vals)) - truth) < 0.02  # DR stays on truth


def test_ess_is_full_under_on_policy() -> None:
    logging = np.array([[0.4, 0.6], [0.7, 0.3], [0.5, 0.5]], dtype=np.float64)
    sim = BanditSimulator(CONTEXT_PROBS, REWARD_MEANS, logging)
    # Target identical to the logging policy => weights all 1 => ESS == n.
    ev = evaluate_policy(sim.sample(1000, seed=5), TabularPolicy(logging), OPE)
    assert ev.ess_fraction > 0.999
