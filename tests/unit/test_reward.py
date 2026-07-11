"""Unit tests for the composite routing reward."""

from __future__ import annotations

import pytest

from specialist_router.config import RewardConfig
from specialist_router.router.reward import compute_reward, normalize_cost, normalize_latency

CFG = RewardConfig(lambda_cost=0.3, mu_latency=0.1, cost_ref_usd=0.02, latency_ref_s=8.0)


def test_perfect_free_instant_answer_scores_one() -> None:
    breakdown = compute_reward(1, cost_usd=0.0, latency_s=0.0, config=CFG)
    assert breakdown.reward == pytest.approx(1.0)
    assert breakdown.cost_norm == 0.0
    assert breakdown.latency_norm == 0.0


def test_penalties_subtract_at_reference_scale() -> None:
    # Cost and latency exactly at reference => normalized to 1 => full lambda/mu penalty.
    breakdown = compute_reward(1, cost_usd=0.02, latency_s=8.0, config=CFG)
    assert breakdown.cost_norm == 1.0
    assert breakdown.latency_norm == 1.0
    assert breakdown.reward == pytest.approx(1.0 - 0.3 - 0.1)


def test_normalization_clamps_above_reference() -> None:
    assert normalize_cost(1.0, CFG) == 1.0  # far above cost_ref
    assert normalize_latency(100.0, CFG) == 1.0
    assert normalize_cost(0.01, CFG) == pytest.approx(0.5)


def test_incorrect_answer_can_go_negative() -> None:
    breakdown = compute_reward(0, cost_usd=0.02, latency_s=8.0, config=CFG)
    assert breakdown.reward == pytest.approx(-0.4)


@pytest.mark.parametrize("bad_quality", [-1, 2, 5])
def test_quality_must_be_binary(bad_quality: int) -> None:
    with pytest.raises(ValueError, match="quality must be 0 or 1"):
        compute_reward(bad_quality, 0.0, 0.0, CFG)


def test_negative_cost_or_latency_rejected() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        compute_reward(1, -0.01, 0.0, CFG)
    with pytest.raises(ValueError, match="latency_s"):
        compute_reward(1, 0.0, -1.0, CFG)
