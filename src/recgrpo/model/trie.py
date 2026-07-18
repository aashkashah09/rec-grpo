"""Prefix trie over the tokenized catalog.

Decoding is masked to the trie at every step, so a generated sequence is always
a real item and the model never has to spend probability mass on the 99.9% of
the code space that is empty. Adding an item is a trie insert: a new listing
becomes recommendable as soon as its text is tokenized, with no retraining.
"""

from __future__ import annotations

import numpy as np
import torch

from ..semid.vocab import EOS, VOCAB_SIZE


class CatalogTrie:
    def __init__(self, item_tokens: np.ndarray | None = None, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.children: list[dict[int, int]] = [{}]
        self.item_at: dict[int, int] = {}
        self.code_len = 0
        self._mask_cache: dict[int, torch.Tensor] = {}
        if item_tokens is not None:
            self.build(item_tokens)

    def build(self, item_tokens: np.ndarray) -> "CatalogTrie":
        self.children = [{}]
        self.item_at = {}
        self.code_len = int(item_tokens.shape[1])
        for item, tokens in enumerate(item_tokens):
            self.add_item(item, tokens)
        return self

    def add_item(self, item: int, tokens: np.ndarray | list[int]) -> None:
        node = 0
        for token in tokens:
            token = int(token)
            nxt = self.children[node].get(token)
            if nxt is None:
                nxt = len(self.children)
                self.children.append({})
                self.children[node][token] = nxt
            node = nxt
        if EOS not in self.children[node]:
            leaf = len(self.children)
            self.children.append({})
            self.children[node][EOS] = leaf
        self.item_at[self.children[node][EOS]] = item
        self._mask_cache.clear()

    def root(self) -> int:
        return 0

    def step(self, node: int, token: int) -> int | None:
        return self.children[node].get(int(token))

    def allowed(self, node: int) -> list[int]:
        return list(self.children[node].keys())

    def item_for(self, node: int) -> int | None:
        return self.item_at.get(node)

    def is_leaf(self, node: int) -> bool:
        return node in self.item_at

    def mask(self, node: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Additive log-space mask: 0 for permitted tokens, -inf elsewhere."""
        cached = self._mask_cache.get(node)
        if cached is None:
            cached = torch.full((self.vocab_size,), float("-inf"))
            keys = torch.tensor(list(self.children[node].keys()), dtype=torch.long)
            cached[keys] = 0.0
            self._mask_cache[node] = cached
        return cached.to(device)

    def masks_for(self, nodes: list[int], device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.stack([self.mask(n, device) for n in nodes])

    def __len__(self) -> int:
        return len(self.item_at)
