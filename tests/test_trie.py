import numpy as np
import torch

from recgrpo.model.trie import CatalogTrie
from recgrpo.semid.assign import assign_semantic_ids
from recgrpo.semid.vocab import EOS


def test_every_item_is_reachable(trie, item_tokens):
    assert len(trie) == len(item_tokens)
    for item, tokens in enumerate(item_tokens):
        node = trie.root()
        for token in tokens:
            node = trie.step(node, int(token))
            assert node is not None
        assert trie.item_for(trie.step(node, EOS)) == item


def test_mask_permits_exactly_the_trie_children(trie):
    mask = trie.mask(trie.root())
    permitted = set(torch.isfinite(mask).nonzero(as_tuple=True)[0].tolist())
    assert permitted == set(trie.allowed(trie.root()))


def test_off_catalog_prefixes_are_dead_ends(trie):
    assert trie.step(trie.root(), 3) is None


def test_new_item_becomes_reachable_without_a_rebuild():
    codes = np.array([[1, 2, 3], [4, 5, 6]])
    tokens = assign_semantic_ids(codes)
    trie = CatalogTrie(tokens)
    assert len(trie) == 2

    fresh = assign_semantic_ids(np.array([[7, 8, 9]]))[0]
    trie.add_item(2, fresh)

    node = trie.root()
    for token in fresh:
        node = trie.step(node, int(token))
    assert trie.item_for(trie.step(node, EOS)) == 2
    assert len(trie) == 3


def test_mask_cache_is_invalidated_on_insert():
    trie = CatalogTrie(assign_semantic_ids(np.array([[1, 2, 3]])))
    before = int(torch.isfinite(trie.mask(trie.root())).sum())
    trie.add_item(1, assign_semantic_ids(np.array([[9, 2, 3]]))[0])
    assert int(torch.isfinite(trie.mask(trie.root())).sum()) == before + 1
