"""Turning RQ-VAE codes into unique per-item token sequences."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .vocab import CODEBOOK_SIZE, DEDUP_CAPACITY, N_LEVELS, codes_to_tokens


def assign_semantic_ids(codes: np.ndarray, capacity: int = DEDUP_CAPACITY) -> np.ndarray:
    """(n_items, n_levels) codes -> (n_items, n_levels + 1) vocabulary tokens.

    Items sharing all three codes are separated by a suffix assigned in item
    index order, so the mapping is a pure function of the codes and does not
    depend on how the catalog was shuffled.
    """
    seen: Counter[tuple[int, ...]] = Counter()
    suffix = np.zeros(len(codes), dtype=np.int64)
    for i, row in enumerate(codes):
        key = tuple(int(c) for c in row)
        suffix[i] = seen[key]
        seen[key] += 1
    overflow = int((suffix >= capacity).sum())
    if overflow:
        raise ValueError(
            f"{overflow} items exceed the suffix capacity of {capacity}; "
            "increase dedup_capacity or retrain the quantizer"
        )
    full = np.concatenate([codes.astype(np.int64), suffix[:, None]], axis=1)
    return codes_to_tokens(full)


def semantic_id_stats(codes: np.ndarray) -> dict:
    """Codebook occupancy and collision structure, reported after assignment."""
    groups: dict[tuple[int, ...], int] = defaultdict(int)
    for row in codes:
        groups[tuple(int(c) for c in row)] += 1
    sizes = Counter(groups.values())
    colliding_items = sum(size * count for size, count in sizes.items() if size > 1)
    return {
        "n_items": int(len(codes)),
        "unique_prefixes": len(groups),
        "colliding_items": colliding_items,
        "max_collision_group": max(groups.values()),
        "collision_group_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "active_codes": [int(len(set(codes[:, level].tolist()))) for level in range(N_LEVELS)],
        "codebook_size": CODEBOOK_SIZE,
    }
