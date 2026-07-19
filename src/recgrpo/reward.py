"""Slate reward.

The signal is the user's own subsequent behaviour in the held-out window: a
slate is scored by how well it ranks the items that user actually went on to
engage with, plus a bonus for surfacing a cold item they engaged with. There is
no learned reward model, so nothing here can drift away from what it is scoring.

The bonus is capped. Without a cap the optimum is a slate of ten cold items,
which pays for itself in warm accuracy long before it stops paying in cold
recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np


def dcg(slate: list[int], targets: set[int], k: int) -> float:
    return sum(1.0 / log2(rank + 2) for rank, item in enumerate(slate[:k]) if item in targets)


def ideal_dcg(n_targets: int, k: int) -> float:
    return sum(1.0 / log2(rank + 2) for rank in range(min(n_targets, k)))


def ndcg(slate: list[int], targets: set[int], k: int = 10) -> float:
    if not targets:
        return 0.0
    ideal = ideal_dcg(len(targets), k)
    return dcg(slate, targets, k) / ideal if ideal > 0 else 0.0


@dataclass
class RewardBreakdown:
    total: float
    ndcg: float
    cold_bonus: float
    cold_hits: int


class SlateReward:
    """reward = NDCG@k(slate, held-out items) + lambda * capped cold-hit bonus."""

    def __init__(
        self,
        cold_mask: np.ndarray,
        cold_lambda: float = 1.0,
        cold_cap: int = 2,
        metric_k: int = 10,
    ):
        self.cold_mask = cold_mask
        self.cold_lambda = cold_lambda
        self.cold_cap = int(cold_cap)  # zero or negative removes the cap
        self.metric_k = metric_k

    @property
    def capped(self) -> bool:
        return self.cold_cap > 0

    def __call__(self, slate: list[int], targets: set[int]) -> RewardBreakdown:
        relevance = ndcg(slate, targets, self.metric_k)
        cold_hits = sum(
            1 for item in slate[: self.metric_k] if item in targets and self.cold_mask[item]
        )
        # capped: the bonus tops out at cold_cap hits and is normalised to [0, 1],
        # so a slate cannot buy unbounded reward by crowding out warm items.
        # uncapped: every cold hit keeps paying, which is the variant the cap exists to avoid.
        bonus = (
            min(cold_hits, self.cold_cap) / self.cold_cap if self.capped else float(cold_hits)
        )
        return RewardBreakdown(
            total=relevance + self.cold_lambda * bonus,
            ndcg=relevance,
            cold_bonus=bonus,
            cold_hits=cold_hits,
        )

    def batch(self, slates: list[list[int]], targets: set[int]) -> list[RewardBreakdown]:
        return [self(slate, targets) for slate in slates]


def group_advantages(rewards: np.ndarray, normalize: bool = True, eps: float = 1e-4) -> np.ndarray:
    """Group-relative advantages: each rollout is scored against its own group.

    With a degenerate group -- every slate earning the same reward, which happens
    often once a user's held-out set is empty of anything the policy can reach --
    the advantage is zero and the group contributes no gradient.
    """
    centred = rewards - rewards.mean()
    if not normalize:
        return centred
    return centred / (rewards.std() + eps)
