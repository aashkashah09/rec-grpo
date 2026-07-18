from pathlib import Path

import numpy as np
import torch
import yaml

from recgrpo.model.generate import beam_search, sample_slate, sequence_logprobs
from recgrpo.model.transformer import SemanticIDTransformer, count_parameters
from recgrpo.semid.vocab import BOS, SEP, VOCAB_SIZE


def tiny_model(**kwargs) -> SemanticIDTransformer:
    defaults = dict(d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=64, dropout=0.0)
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SemanticIDTransformer(**defaults).eval()


def test_forward_shapes_and_padding_is_ignored():
    model = tiny_model()
    tokens = torch.randint(4, VOCAB_SIZE, (3, 12))
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[1, 8:] = False

    logits = model(tokens, attn_mask=mask)
    assert logits.shape == (3, 12, VOCAB_SIZE)
    assert torch.isfinite(logits).all()

    padded = tokens.clone()
    padded[1, 8:] = 0
    other = model(padded, attn_mask=mask)
    torch.testing.assert_close(logits[1, :8], other[1, :8], atol=1e-5, rtol=1e-5)


def test_attention_is_causal():
    model = tiny_model()
    tokens = torch.randint(4, VOCAB_SIZE, (1, 10))
    base = model(tokens)
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % VOCAB_SIZE
    torch.testing.assert_close(base[:, :-1], model(changed)[:, :-1], atol=1e-6, rtol=1e-6)


def test_tied_embeddings_are_counted_once():
    tied = count_parameters(tiny_model(tie_embeddings=True))
    untied = count_parameters(tiny_model(tie_embeddings=False))
    assert untied - tied == VOCAB_SIZE * 32


def test_beam_search_only_returns_catalog_items(trie, item_tokens):
    model = tiny_model()
    prompt = torch.tensor([BOS, *item_tokens[3], SEP])
    items, scores = beam_search(model, prompt, trie, beam_size=16, topk=10)

    assert len(items) == 10
    assert len(set(items)) == 10
    assert all(0 <= item < len(item_tokens) for item in items)
    assert scores == sorted(scores, reverse=True)


def test_sampled_slates_are_valid_and_scored(trie, item_tokens):
    model = tiny_model()
    prompt = torch.tensor([BOS, *item_tokens[0], SEP])
    items, sequences = sample_slate(model, prompt, trie, slate_size=10, oversample=24)

    assert len(items) == 10
    assert len(set(items)) == 10
    assert sequences.shape == (10, trie.code_len + 1)
    for item, sequence in zip(items, sequences):
        assert np.array_equal(sequence[:-1].numpy(), item_tokens[item])


def test_sequence_logprobs_are_masked_to_the_trie(trie, item_tokens):
    model = tiny_model()
    prompt = torch.tensor([BOS, SEP])
    sequences = torch.as_tensor(
        np.c_[item_tokens[:4], np.full(4, 2)], dtype=torch.long
    )
    logprobs = sequence_logprobs(model, prompt, sequences, trie)

    assert logprobs.shape == (4, trie.code_len + 1)
    assert torch.isfinite(logprobs).all()
    assert (logprobs <= 0).all()


def test_builds_and_runs_from_the_shipped_config():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/sft.yaml").read_text())["model"]
    assert cfg["d_model"] % cfg["n_heads"] == 0

    model = SemanticIDTransformer(**cfg).eval()
    tokens = torch.randint(4, VOCAB_SIZE, (2, 48))
    logits = model(tokens, attn_mask=torch.ones(2, 48, dtype=torch.bool))

    assert logits.shape == (2, 48, VOCAB_SIZE)
    assert torch.isfinite(logits).all()
    assert model.head.weight is model.tok_emb.weight
    assert len(model.blocks) == cfg["n_layers"]
