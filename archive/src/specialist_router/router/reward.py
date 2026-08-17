"""The composite routing reward: ``reward = quality − λ·cost_norm − μ·latency_norm``.

Pure functions only, so the reward is trivially testable and every logged decision's reward is
recomputable from its components. Two deliberate design choices (ADR-007):

* **Quality is the verifier's binary verdict** (``1`` correct, ``0`` not). Phase 2 uses no
  LLM-judge — every task has deterministic ground truth. The seam where a judge *would* attach
  is documented in :mod:`specialist_router.router` but intentionally left unimplemented.
* **Normalization uses fixed reference scales** from config (``cost_ref_usd``, ``latency_ref_s``),
  not dataset min/max. This keeps the reward stationary across logs and prevents leakage across
  the OPE train / replay split. ``cost_norm`` and ``latency_norm`` are clamped to ``[0, 1]``, so
  λ and μ are exactly the worst-case penalties (in quality units).
"""

from __future__ import annotations

from dataclasses import dataclass

from specialist_router.config import RewardConfig


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """The composite reward and every component that produced it (all logged)."""

    quality: int
    cost_usd: float
    latency_s: float
    cost_norm: float
    latency_norm: float
    reward: float


def normalize_cost(cost_usd: float, config: RewardConfig) -> float:
    """Normalize a per-decision USD cost to ``[0, 1]`` against the fixed reference scale."""
    if cost_usd < 0.0:
        raise ValueError(f"cost_usd must be non-negative, got {cost_usd}")
    return min(cost_usd / config.cost_ref_usd, 1.0)


def normalize_latency(latency_s: float, config: RewardConfig) -> float:
    """Normalize a per-decision latency (seconds) to ``[0, 1]`` against the fixed reference."""
    if latency_s < 0.0:
        raise ValueError(f"latency_s must be non-negative, got {latency_s}")
    return min(latency_s / config.latency_ref_s, 1.0)


def compute_reward(
    quality: int, cost_usd: float, latency_s: float, config: RewardConfig
) -> RewardBreakdown:
    """Compute the composite reward and its components.

    Args:
        quality: The verifier verdict as ``1`` (correct) or ``0`` (incorrect).
        cost_usd: The decision's realized cost in USD (``≥ 0``).
        latency_s: The decision's realized latency in seconds (``≥ 0``).
        config: Reward weights and fixed reference scales.

    Returns:
        A :class:`RewardBreakdown` with the normalized components and the composite reward.

    Raises:
        ValueError: If ``quality`` is not 0/1 or a cost/latency is negative.
    """
    if quality not in (0, 1):
        raise ValueError(f"quality must be 0 or 1, got {quality!r}")
    cost_norm = normalize_cost(cost_usd, config)
    latency_norm = normalize_latency(latency_s, config)
    reward = quality - config.lambda_cost * cost_norm - config.mu_latency * latency_norm
    return RewardBreakdown(
        quality=quality,
        cost_usd=cost_usd,
        latency_s=latency_s,
        cost_norm=cost_norm,
        latency_norm=latency_norm,
        reward=reward,
    )
