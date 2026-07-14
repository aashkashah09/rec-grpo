"""Frozen sentence encoder over item text.

Nothing here is trained. The encoder is the only place item content enters the
system, and it never sees interactions, which is what makes a brand new item
representable the moment it is listed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def encode_items(
    texts: list[str],
    model_name: str = "sentence-transformers/sentence-t5-base",
    batch_size: int = 128,
    normalize: bool = False,
    cache: str | Path | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Return (n_items, dim) float32 embeddings, cached to disk if a path is given."""
    if cache is not None and Path(cache).exists():
        emb = np.load(cache)
        if len(emb) == len(texts):
            logger.info("loaded cached item embeddings from %s %s", cache, emb.shape)
            return emb
        logger.warning("cache %s has %d rows, expected %d -- re-encoding", cache, len(emb), len(texts))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.eval()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    ).astype(np.float32)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        logger.info("wrote item embeddings %s -> %s", emb.shape, cache)
    return emb
