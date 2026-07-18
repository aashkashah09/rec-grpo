import numpy as np
import pytest
import torch

from recgrpo.semid.assign import assign_semantic_ids, semantic_id_stats
from recgrpo.semid.rqvae import RQVAE
from recgrpo.semid.vocab import (
    CODE_LEN,
    DEDUP_OFFSET,
    LEVEL_OFFSETS,
    VOCAB_SIZE,
    codes_to_tokens,
    tokens_to_codes,
)


def test_token_blocks_are_disjoint():
    assert LEVEL_OFFSETS == (4, 260, 516)
    assert DEDUP_OFFSET == 772
    assert VOCAB_SIZE == 1028


def test_codes_round_trip():
    codes = np.array([[0, 255, 12, 3], [17, 4, 200, 0]])
    assert np.array_equal(tokens_to_codes(codes_to_tokens(codes)), codes)


def test_assignment_is_unique_under_collisions():
    codes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6]])
    tokens = assign_semantic_ids(codes)
    assert tokens.shape == (4, CODE_LEN)
    assert len({tuple(row) for row in tokens}) == 4
    # the suffix is what separates them; the learned codes are untouched
    assert np.array_equal(tokens[:3, :3], codes_to_tokens(np.c_[codes[:3], np.zeros(3, int)])[:, :3])
    assert tokens[:3, -1].tolist() == [DEDUP_OFFSET, DEDUP_OFFSET + 1, DEDUP_OFFSET + 2]


def test_assignment_rejects_suffix_overflow():
    codes = np.zeros((5, 3), dtype=int)
    with pytest.raises(ValueError, match="suffix capacity"):
        assign_semantic_ids(codes, capacity=4)


def test_stats_report_collision_structure():
    codes = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]])
    stats = semantic_id_stats(codes)
    assert stats["n_items"] == 3
    assert stats["unique_prefixes"] == 2
    assert stats["colliding_items"] == 2
    assert stats["max_collision_group"] == 2


def test_rqvae_forward_and_code_shapes():
    torch.manual_seed(0)
    model = RQVAE(input_dim=32, hidden_dims=[16], latent_dim=8, codebook_size=16)
    x = torch.randn(24, 32)
    out = model(x)
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (24, 3)
    assert out["codes"].max() < 16
    assert torch.isfinite(out["loss"])

    codes = model.encode_codes(x)
    assert codes.shape == (24, 3)


def test_residual_quantization_reduces_the_residual():
    torch.manual_seed(0)
    model = RQVAE(input_dim=32, hidden_dims=[16], latent_dim=8, codebook_size=64)
    x = torch.randn(256, 32)
    with torch.no_grad():
        z = model.encoder(x)
        model.quantizer.init_from_data(z, iters=10)
        residual = z
        errors = []
        for level in model.quantizer.levels:
            q, _, _ = level(residual)
            residual = residual - q
            errors.append(float(residual.pow(2).mean()))
    assert errors == sorted(errors, reverse=True)
