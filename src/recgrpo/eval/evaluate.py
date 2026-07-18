"""Full-catalog evaluation of a generative retriever."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..data.dataset import EvalExample, encode_prompt
from ..model.generate import beam_search
from ..model.trie import CatalogTrie
from .metrics import RankingMetrics


@torch.no_grad()
def generate_slates(
    model,
    examples: list[EvalExample],
    item_tokens: np.ndarray,
    trie: CatalogTrie,
    beam_size: int = 30,
    slate_size: int = 20,
    device=None,
    desc: str = "eval",
) -> list[list[int]]:
    device = device or next(model.parameters()).device
    model.eval()
    codes = torch.as_tensor(item_tokens, dtype=torch.long)

    slates = []
    for example in tqdm(examples, desc=desc, unit="ex"):
        prompt = encode_prompt(example.history, codes)
        items, _ = beam_search(model, prompt, trie, beam_size=beam_size, topk=slate_size, device=device)
        slates.append(items)
    return slates


def score_slates(
    slates: list[list[int]],
    examples: list[EvalExample],
    cold_mask: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    extra_slices: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Overall metrics plus the cold slice and any further item-level slices."""
    metrics = RankingMetrics(ks=ks)
    for slate, example in zip(slates, examples):
        slices = []
        if cold_mask[example.target]:
            slices.append("cold")
            for name, mask in (extra_slices or {}).items():
                if mask[example.target]:
                    slices.append(name)
        metrics.update(slate, example.target, tuple(slices))
    return metrics.result()


def evaluate(
    model,
    examples: list[EvalExample],
    item_tokens: np.ndarray,
    trie: CatalogTrie,
    cold_mask: np.ndarray,
    beam_size: int = 30,
    ks: tuple[int, ...] = (5, 10, 20),
    extra_slices: dict[str, np.ndarray] | None = None,
    device=None,
    desc: str = "eval",
) -> tuple[dict[str, dict[str, float]], list[list[int]]]:
    slates = generate_slates(
        model,
        examples,
        item_tokens,
        trie,
        beam_size=beam_size,
        slate_size=max(ks),
        device=device,
        desc=desc,
    )
    return score_slates(slates, examples, cold_mask, ks=ks, extra_slices=extra_slices), slates
