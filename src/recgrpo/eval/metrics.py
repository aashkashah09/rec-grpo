"""Ranking metrics over full-catalog slates.

Each evaluation example has exactly one target -- the interaction being
predicted -- so Recall@k is the hit rate and NDCG@k is 1/log2(rank+2) at the
position the target lands in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log2


def recall_at_k(slate: list[int], target: int, k: int) -> float:
    return float(target in slate[:k])


def ndcg_at_k(slate: list[int], target: int, k: int) -> float:
    for rank, item in enumerate(slate[:k]):
        if item == target:
            return 1.0 / log2(rank + 2)
    return 0.0


@dataclass
class RankingMetrics:
    """Accumulates hits over a stream of (slate, target) pairs.

    Slices are named subsets of the examples -- 'cold' for targets with no
    training interaction, and whatever else the caller passes -- scored on the
    same slates so the numbers are directly comparable.
    """

    ks: tuple[int, ...] = (5, 10, 20)
    _hits: dict[str, dict[int, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    _dcg: dict[str, dict[int, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    _n: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, slate: list[int], target: int, slices: tuple[str, ...] = ()) -> None:
        for name in ("all", *slices):
            self._n[name] += 1
            for k in self.ks:
                self._hits[name][k] += recall_at_k(slate, target, k)
                self._dcg[name][k] += ndcg_at_k(slate, target, k)

    def result(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, n in self._n.items():
            if n == 0:
                continue
            row: dict[str, float] = {"n": n}
            for k in self.ks:
                row[f"recall@{k}"] = self._hits[name][k] / n
                row[f"ndcg@{k}"] = self._dcg[name][k] / n
                row[f"hits@{k}"] = int(self._hits[name][k])
            out[name] = row
        return out
