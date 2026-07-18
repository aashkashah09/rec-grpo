import numpy as np
import pytest

from recgrpo.semid.assign import assign_semantic_ids
from recgrpo.model.trie import CatalogTrie


@pytest.fixture
def item_tokens() -> np.ndarray:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(64, 3))
    return assign_semantic_ids(codes)


@pytest.fixture
def trie(item_tokens) -> CatalogTrie:
    return CatalogTrie(item_tokens)
