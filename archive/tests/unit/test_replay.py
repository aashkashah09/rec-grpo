"""Unit tests for replay-A/B calibration and the reward-scale parity guard."""

from __future__ import annotations

import pytest

from specialist_router.ope.ci import ConfidenceInterval
from specialist_router.ope.replay import (
    ReplayScaleError,
    calibrate,
    realized_value,
    reward_signature,
)
from tests.support import make_decision


def test_reward_signature_unique() -> None:
    decisions = [make_decision(decision_id=f"d{i}") for i in range(3)]
    assert reward_signature(decisions) == (0.3, 0.1, 0.02, 8.0)


def test_reward_signature_rejects_mixed() -> None:
    decisions = [make_decision(reward_lambda=0.3), make_decision(reward_lambda=0.5)]
    with pytest.raises(ReplayScaleError, match="mix"):
        reward_signature(decisions)


def test_realized_value_is_mean_reward() -> None:
    decisions = [make_decision(reward=0.2), make_decision(reward=0.8)]
    assert realized_value(decisions) == pytest.approx(0.5)


def test_calibrate_inside_ci() -> None:
    logged = [make_decision(decision_id=f"l{i}") for i in range(4)]
    replay = [make_decision(decision_id=f"r{i}", reward=0.5) for i in range(4)]
    ci = ConfidenceInterval(point=0.52, lo=0.45, hi=0.60)
    cal = calibrate("uniform", logged, replay, ci, "doubly_robust")
    assert cal.realized_value == pytest.approx(0.5)
    assert cal.inside_ci is True
    assert cal.abs_error == pytest.approx(0.02)


def test_calibrate_enforces_reward_scale_parity() -> None:
    # The rider: replay must use the SAME lambda/mu as the OPE it validates, asserted in code.
    logged = [make_decision(reward_lambda=0.3, reward_mu=0.1)]
    replay = [make_decision(reward_lambda=0.5, reward_mu=0.1, reward=0.5)]
    ci = ConfidenceInterval(point=0.5, lo=0.4, hi=0.6)
    with pytest.raises(ReplayScaleError, match="does not match"):
        calibrate("uniform", logged, replay, ci, "doubly_robust")
