"""Replay-A/B validation: does the OPE prediction match the value the policy actually realizes?

An OPE number is only trustworthy if, when we *deploy* the candidate policy on fresh traffic, the
realized average reward lands where OPE said it would. This module compares the two and produces a
calibration record per policy.

**Reward-scale parity (a hard requirement):** the realized reward and the OPE prediction must be
computed with the *same* reward weights and reference scales — otherwise "predicted 0.7, realized
0.6" could be an artifact of two different reward definitions, not miscalibration. Every
:class:`~specialist_router.env.records.RouterDecision` records its ``reward_lambda``/``reward_mu``/
``cost_ref_usd``/``latency_ref_s``; :func:`calibrate` asserts the logged set and the replay set
share one signature and that the two signatures match, raising :class:`ReplayScaleError` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specialist_router.env.records import RouterDecision
from specialist_router.ope.ci import ConfidenceInterval

RewardSignature = tuple[float, float, float, float]


class ReplayScaleError(Exception):
    """Raised when logged and replay rewards are not on the same (λ, μ, refs) scale."""


def reward_signature(decisions: list[RouterDecision]) -> RewardSignature:
    """Return the single ``(λ, μ, cost_ref, latency_ref)`` shared by all decisions.

    Raises:
        ReplayScaleError: If the list is empty or mixes more than one reward definition.
    """
    if not decisions:
        raise ReplayScaleError("cannot take a reward signature of zero decisions")
    signatures = {
        (d.reward_lambda, d.reward_mu, d.cost_ref_usd, d.latency_ref_s) for d in decisions
    }
    if len(signatures) != 1:
        raise ReplayScaleError(f"decisions mix {len(signatures)} reward definitions: {signatures}")
    return signatures.pop()


def realized_value(decisions: list[RouterDecision]) -> float:
    """The realized mean reward over a set of deployed decisions."""
    if not decisions:
        raise ReplayScaleError("cannot compute realized value of zero decisions")
    return float(np.mean([d.reward for d in decisions]))


@dataclass(frozen=True, slots=True)
class ReplayCalibration:
    """One row of the replay calibration table: OPE prediction vs realized value."""

    policy_name: str
    estimator: str
    ope_point: float
    ope_lo: float
    ope_hi: float
    realized_value: float
    realized_n: int
    inside_ci: bool
    abs_error: float


def calibrate(
    policy_name: str,
    logged_decisions: list[RouterDecision],
    replay_decisions: list[RouterDecision],
    ope_interval: ConfidenceInterval,
    estimator: str,
) -> ReplayCalibration:
    """Compare an OPE prediction to the value realized by deploying the policy on fresh traffic.

    Args:
        policy_name: The candidate policy's name.
        logged_decisions: The offline (Uniform-logged) dataset the OPE was computed on.
        replay_decisions: Fresh decisions from *deploying* the candidate policy.
        ope_interval: The OPE point estimate and CI for this policy (e.g. from DR).
        estimator: Which estimator ``ope_interval`` came from (for the report).

    Returns:
        A :class:`ReplayCalibration` row.

    Raises:
        ReplayScaleError: If the logged and replay rewards are not on the same scale.
    """
    logged_signature = reward_signature(logged_decisions)
    replay_signature = reward_signature(replay_decisions)
    if logged_signature != replay_signature:
        raise ReplayScaleError(
            "replay reward scale does not match the OPE it validates: "
            f"logged {logged_signature} vs replay {replay_signature}. "
            "The realized value and the OPE prediction must share (λ, μ, cost_ref, latency_ref)."
        )
    realized = realized_value(replay_decisions)
    return ReplayCalibration(
        policy_name=policy_name,
        estimator=estimator,
        ope_point=ope_interval.point,
        ope_lo=ope_interval.lo,
        ope_hi=ope_interval.hi,
        realized_value=realized,
        realized_n=len(replay_decisions),
        inside_ci=ope_interval.lo <= realized <= ope_interval.hi,
        abs_error=abs(realized - ope_interval.point),
    )
