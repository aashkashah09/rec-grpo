"""What the policy puts in its slates, and whether the reward could have taught it to.

Recall alone cannot separate "the policy learned to surface cold items" from
"the policy memorised the cold items the reward paid out on". This splits every
cold placement by whether that item was ever scored by the reward during
post-training. If the shift were memorisation, only the reward-visible half
would move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..data.dataset import EvalExample
from .bootstrap import bootstrap_difference


@dataclass
class SlateComposition:
    model: str
    n_examples: int
    n_positions: int
    cold_positions: int
    reward_visible_positions: int
    never_rewarded_positions: int
    cold_share: float
    reward_visible_share: float
    never_rewarded_share: float
    cold_hits: int
    reward_visible_hits: int
    never_rewarded_hits: int
    cold_conversion: float

    def to_dict(self) -> dict:
        return asdict(self)


def composition(
    name: str,
    slates: list[list[int]],
    examples: list[EvalExample],
    cold_mask: np.ndarray,
    reward_visible: np.ndarray,
    k: int = 10,
) -> SlateComposition:
    """Count cold slate positions and how many of them converted into a hit."""
    n_positions = len(slates) * k
    cold = visible = never = 0
    cold_hits = visible_hits = never_hits = 0

    for slate, example in zip(slates, examples):
        for item in slate[:k]:
            if not cold_mask[item]:
                continue
            cold += 1
            hit = item == example.target
            cold_hits += hit
            if reward_visible[item]:
                visible += 1
                visible_hits += hit
            else:
                never += 1
                never_hits += hit

    return SlateComposition(
        model=name,
        n_examples=len(slates),
        n_positions=n_positions,
        cold_positions=cold,
        reward_visible_positions=visible,
        never_rewarded_positions=never,
        cold_share=cold / n_positions,
        reward_visible_share=visible / n_positions,
        never_rewarded_share=never / n_positions,
        cold_hits=int(cold_hits),
        reward_visible_hits=int(visible_hits),
        never_rewarded_hits=int(never_hits),
        cold_conversion=(cold_hits / cold) if cold else 0.0,
    )


def per_example_cold_positions(
    slates: list[list[int]],
    cold_mask: np.ndarray,
    reward_visible: np.ndarray,
    k: int = 10,
    subset: str = "never_rewarded",
) -> np.ndarray:
    """Per-example count of cold placements, as the unit the bootstrap resamples."""
    counts = np.zeros(len(slates))
    for i, slate in enumerate(slates):
        for item in slate[:k]:
            if not cold_mask[item]:
                continue
            if subset == "cold":
                counts[i] += 1
            elif subset == "reward_visible" and reward_visible[item]:
                counts[i] += 1
            elif subset == "never_rewarded" and not reward_visible[item]:
                counts[i] += 1
    return counts / k


def compare(
    baseline_slates: list[list[int]],
    policy_slates: list[list[int]],
    examples: list[EvalExample],
    cold_mask: np.ndarray,
    reward_visible: np.ndarray,
    k: int = 10,
    subset: str = "never_rewarded",
    n_replicates: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap the change in cold slate share between two policies."""
    users = np.array([ex.user for ex in examples])
    a = per_example_cold_positions(baseline_slates, cold_mask, reward_visible, k, subset)
    b = per_example_cold_positions(policy_slates, cold_mask, reward_visible, k, subset)
    stats = bootstrap_difference(a, b, users, n_replicates=n_replicates, seed=seed)
    stats["subset"] = subset
    stats["baseline_share"] = float(a.mean())
    stats["policy_share"] = float(b.mean())
    stats["relative_change"] = float(b.mean() / a.mean() - 1.0) if a.mean() else float("nan")
    return stats
