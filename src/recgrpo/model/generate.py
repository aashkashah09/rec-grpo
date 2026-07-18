"""Trie-constrained decoding.

Prompts in a batch have different lengths, and the model uses absolute
positions, so sequences are right-padded and each row keeps its own write
cursor. Every step masks the logits to the tokens the trie allows from the
row's current node, which is what guarantees a decode lands on a catalog item.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..semid.vocab import PAD
from .trie import CatalogTrie


def _pack(prompts: list[torch.Tensor], width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    buf = torch.full((len(prompts), width), PAD, dtype=torch.long, device=device)
    lengths = torch.empty(len(prompts), dtype=torch.long, device=device)
    for i, prompt in enumerate(prompts):
        n = len(prompt)
        buf[i, :n] = prompt.to(device)
        lengths[i] = n
    return buf, lengths


def _next_logits(model, buf: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Logits at each row's write cursor."""
    width = int(lengths.max())
    view = buf[:, :width]
    attn_mask = torch.arange(width, device=buf.device)[None] < lengths[:, None]
    logits = model(view, attn_mask=attn_mask)
    return logits[torch.arange(len(buf), device=buf.device), lengths - 1]


@torch.no_grad()
def beam_search(
    model,
    prompt: torch.Tensor,
    trie: CatalogTrie,
    beam_size: int = 30,
    topk: int = 10,
    device=None,
) -> tuple[list[int], list[float]]:
    """Return the top-`topk` catalog items for one prompt, with their log-probs."""
    device = device or next(model.parameters()).device
    steps = trie.code_len + 1
    width = len(prompt) + steps

    buf, lengths = _pack([prompt], width, device)
    scores = torch.zeros(1, device=device)
    nodes = [trie.root()]

    for _ in range(steps):
        logits = _next_logits(model, buf, lengths)
        logprobs = F.log_softmax(logits.float(), dim=-1) + trie.masks_for(nodes, device)
        flat = (scores[:, None] + logprobs).reshape(-1)

        k = min(beam_size, int(torch.isfinite(flat).sum()))
        scores, flat_idx = flat.topk(k)
        beam_idx = torch.div(flat_idx, logprobs.shape[-1], rounding_mode="floor")
        tokens = flat_idx % logprobs.shape[-1]

        buf = buf[beam_idx].clone()
        lengths = lengths[beam_idx].clone()
        buf[torch.arange(k, device=device), lengths] = tokens
        lengths += 1
        nodes = [trie.step(nodes[int(b)], int(t)) for b, t in zip(beam_idx, tokens)]

    items, item_scores = [], []
    for node, score in zip(nodes, scores.tolist()):
        item = trie.item_for(node)
        if item is not None and item not in items:
            items.append(item)
            item_scores.append(score)
        if len(items) == topk:
            break
    return items, item_scores


@torch.no_grad()
def sample_slate(
    model,
    prompt: torch.Tensor,
    trie: CatalogTrie,
    slate_size: int = 10,
    oversample: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    beam_backfill: bool = True,
    device=None,
) -> tuple[list[int], torch.Tensor]:
    """Draw a slate by sampling under the trie, then ranking by sequence log-prob.

    Returns the items and their token sequences, ordered by log-prob. Sampling
    with replacement and deduplicating keeps the slate a valid sample from the
    policy while still producing distinct recommendations.
    """
    device = device or next(model.parameters()).device
    steps = trie.code_len + 1
    width = len(prompt) + steps

    buf, lengths = _pack([prompt] * oversample, width, device)
    nodes = [trie.root()] * oversample
    scores = torch.zeros(oversample, device=device)

    for _ in range(steps):
        logits = _next_logits(model, buf, lengths)
        logprobs = F.log_softmax(logits.float() / temperature, dim=-1) + trie.masks_for(nodes, device)
        if top_k:
            cutoff = logprobs.topk(min(top_k, logprobs.shape[-1]), dim=-1).values[:, -1:]
            logprobs = logprobs.masked_fill(logprobs < cutoff, float("-inf"))
        probs = logprobs.exp()
        tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        scores += logprobs.gather(1, tokens[:, None]).squeeze(1)
        buf[torch.arange(oversample, device=device), lengths] = tokens
        lengths += 1
        nodes = [trie.step(n, int(t)) for n, t in zip(nodes, tokens)]

    order = scores.argsort(descending=True)
    items: list[int] = []
    rows: list[int] = []
    for i in order.tolist():
        item = trie.item_for(nodes[i])
        if item is not None and item not in items:
            items.append(item)
            rows.append(i)
        if len(items) == slate_size:
            break

    start = len(prompt)
    sequences = buf[rows, start : start + steps] if rows else torch.empty(0, steps, dtype=torch.long)

    if beam_backfill and len(items) < slate_size:
        extra_items, _ = beam_search(model, prompt, trie, beam_size=4 * slate_size, topk=slate_size * 2)
        pad_rows = []
        for item in extra_items:
            if len(items) == slate_size:
                break
            if item in items:
                continue
            items.append(item)
            pad_rows.append(item)
        if pad_rows:
            item_tokens = torch.stack(
                [_tokens_for_item(trie, item).to(device) for item in pad_rows]
            )
            sequences = torch.cat([sequences.to(device), item_tokens], dim=0)

    return items, sequences


def _tokens_for_item(trie: CatalogTrie, item: int) -> torch.Tensor:
    """Walk the trie back out to the token sequence of a known item."""
    if not hasattr(trie, "_reverse"):
        reverse: dict[int, list[int]] = {}
        stack: list[tuple[int, list[int]]] = [(trie.root(), [])]
        while stack:
            node, path = stack.pop()
            leaf_item = trie.item_for(node)
            if leaf_item is not None:
                reverse[leaf_item] = path
            for token, child in trie.children[node].items():
                stack.append((child, path + [token]))
        trie._reverse = reverse  # type: ignore[attr-defined]
    return torch.tensor(trie._reverse[item], dtype=torch.long)  # type: ignore[attr-defined]


def sequence_logprobs(
    model,
    prompt: torch.Tensor,
    sequences: torch.Tensor,
    trie: CatalogTrie,
    device=None,
) -> torch.Tensor:
    """Per-token log-probs of `sequences` continuing `prompt`, under the trie mask.

    Shape (n_sequences, seq_len). Gradients flow, so this is what the policy
    ratio is built from.
    """
    device = device or next(model.parameters()).device
    n, steps = sequences.shape
    full = torch.cat([prompt.to(device)[None].expand(n, -1), sequences.to(device)], dim=1)
    attn_mask = torch.ones_like(full, dtype=torch.bool)
    logits = model(full, attn_mask=attn_mask)

    start = len(prompt)
    step_logits = logits[:, start - 1 : start + steps - 1]

    masks = torch.empty(n, steps, trie.vocab_size, device=device)
    for i in range(n):
        node = trie.root()
        for s in range(steps):
            masks[i, s] = trie.mask(node, device)
            node = trie.step(node, int(sequences[i, s]))
            if node is None:  # pragma: no cover - only reachable on a stale trie
                masks[i, s + 1 :] = 0.0
                break

    logprobs = F.log_softmax(step_logits.float(), dim=-1) + masks
    return logprobs.gather(2, sequences.to(device)[:, :, None]).squeeze(2)
