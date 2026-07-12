"""Turning the interaction table into token sequences.

A training example is one prefix of a user's history plus the item that
followed it. At evaluation time the prompt is everything that user did strictly
before the target's timestamp, which under a global temporal split can include
interactions from an earlier evaluation block -- those are in the past relative
to the target, and a deployed system would have them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..semid.vocab import BOS, EOS, PAD, SEP


def user_histories(df: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
    """user -> chronological [(item, timestamp), ...]."""
    ordered = df.sort_values(["user", "timestamp", "item"], kind="mergesort")
    histories: dict[int, list[tuple[int, int]]] = {}
    for user, item, ts in zip(ordered["user"], ordered["item"], ordered["timestamp"]):
        histories.setdefault(int(user), []).append((int(item), int(ts)))
    return histories


@dataclass
class EvalExample:
    user: int
    history: list[int]
    target: int
    timestamp: int


def build_eval_examples(
    full: pd.DataFrame,
    block: pd.DataFrame,
    max_history_items: int,
) -> list[EvalExample]:
    """One example per interaction in `block`, prompted with that user's past.

    Users whose first ever interaction lands in the block get an empty history;
    they are scored, not skipped, so the target count matches the block size.
    """
    histories = user_histories(full)
    examples: list[EvalExample] = []
    for user, item, ts in zip(block["user"], block["item"], block["timestamp"]):
        user, item, ts = int(user), int(item), int(ts)
        past = [i for i, t in histories[user] if t < ts]
        examples.append(
            EvalExample(
                user=user,
                history=past[-max_history_items:],
                target=item,
                timestamp=ts,
            )
        )
    return examples


class SequenceDataset(Dataset):
    """Next-item prediction over semantic-ID token sequences.

    Each user contributes one example per position in their history after the
    first, so a user with n interactions in the window yields n-1 examples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_codes: np.ndarray,
        max_history_items: int = 50,
        min_history_items: int = 1,
    ):
        self.item_codes = torch.as_tensor(item_codes, dtype=torch.long)
        self.code_len = self.item_codes.shape[1]
        self.max_history_items = max_history_items

        histories = user_histories(df)
        self.examples: list[tuple[int, list[int], int]] = []
        for user, events in histories.items():
            items = [i for i, _ in events]
            for pos in range(min_history_items, len(items)):
                history = items[max(0, pos - max_history_items) : pos]
                self.examples.append((user, history, items[pos]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _, history, target = self.examples[idx]
        prompt = encode_prompt(history, self.item_codes)
        target_tokens = torch.cat(
            [self.item_codes[target], torch.tensor([EOS], dtype=torch.long)]
        )
        tokens = torch.cat([prompt, target_tokens])
        # loss is taken on the target tokens only; the prompt is context
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        loss_mask[len(prompt) :] = True
        return {"tokens": tokens, "loss_mask": loss_mask}


def encode_prompt(history: list[int], item_codes: torch.Tensor) -> torch.Tensor:
    """[BOS] <item codes...> [SEP] -- generation starts after SEP."""
    parts = [torch.tensor([BOS], dtype=torch.long)]
    if history:
        parts.append(item_codes[torch.as_tensor(history, dtype=torch.long)].reshape(-1))
    parts.append(torch.tensor([SEP], dtype=torch.long))
    return torch.cat(parts)


def collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch."""
    lengths = [len(b["tokens"]) for b in batch]
    width = max(lengths)
    tokens = torch.full((len(batch), width), PAD, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), width), dtype=torch.bool)
    attn_mask = torch.zeros((len(batch), width), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = lengths[i]
        tokens[i, :n] = b["tokens"]
        loss_mask[i, :n] = b["loss_mask"]
        attn_mask[i, :n] = True
    return {"tokens": tokens, "loss_mask": loss_mask, "attn_mask": attn_mask}
