"""Loading the processed artefacts every downstream script needs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data.dataset import EvalExample, build_eval_examples
from .data.split import Split
from .model.trie import CatalogTrie


@dataclass
class Workspace:
    interactions: pd.DataFrame
    split: Split
    item_tokens: np.ndarray
    cold_mask: np.ndarray
    trie: CatalogTrie

    @property
    def n_items(self) -> int:
        return len(self.item_tokens)

    def eval_examples(
        self,
        block: str,
        max_history_items: int = 50,
        subsample: int | None = None,
        seed: int = 42,
    ) -> list[EvalExample]:
        """Examples for one evaluation block.

        Decoding the whole block takes long enough that running it after every
        epoch would cost more than the training itself, so the in-training
        curves take a fixed subsample. Reported numbers use the full block.
        """
        frame = {"val": self.split.val, "test": self.split.test}[block]
        examples = build_eval_examples(self.interactions, frame, max_history_items)
        if subsample and subsample < len(examples):
            rng = np.random.default_rng(seed)
            picks = np.sort(rng.choice(len(examples), size=subsample, replace=False))
            examples = [examples[i] for i in picks]
        return examples


def load_workspace(processed_dir: str | Path = "data/processed") -> Workspace:
    processed_dir = Path(processed_dir)
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    item_tokens = np.load(processed_dir / "item_tokens.npy")
    cold_mask = np.load(processed_dir / "cold_mask.npy")

    split = Split(
        train=train,
        val=val,
        test=test,
        train_end=int(train["timestamp"].max()),
        val_end=int(val["timestamp"].max()),
    )
    return Workspace(
        interactions=interactions,
        split=split,
        item_tokens=item_tokens,
        cold_mask=cold_mask,
        trie=CatalogTrie(item_tokens),
    )
