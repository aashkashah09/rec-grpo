"""Global temporal split.

One timeline for the whole dataset: interactions are ordered by timestamp and
cut into train / validation / test blocks. A user can contribute to all three.
Unlike leave-one-out, nothing in the training window is dated after anything in
the test window, so a model cannot be scored on a target it could have observed
the future of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: int
    val_end: int

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    tie_break: tuple[str, ...] = ("timestamp", "user", "item"),
) -> Split:
    """Cut the global timeline into three blocks by interaction count.

    Amazon timestamps have day resolution, so a pure timestamp threshold would
    leave the block sizes at the mercy of whichever day happens to straddle the
    cut. Ordering by (timestamp, user, item) and cutting by index keeps the
    blocks the requested size and the ordering reproducible.
    """
    ordered = df.sort_values(list(tie_break), kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]

    return Split(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        train_end=int(train["timestamp"].max()),
        val_end=int(val["timestamp"].max()),
    )


def cold_items(split: Split, n_items: int) -> np.ndarray:
    """Boolean mask over item indices: True where the item has no training interaction.

    These are the items an interaction-based embedding has nothing to learn
    from. They are still in the catalog and still reachable by the generative
    model, because their identity comes from their text.
    """
    mask = np.ones(n_items, dtype=bool)
    mask[split.train["item"].unique()] = False
    return mask


def cold_target_count(split: Split, cold_mask: np.ndarray) -> int:
    return int(cold_mask[split.test["item"].to_numpy()].sum())
