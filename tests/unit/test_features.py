"""Unit tests for the context featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from specialist_router.config import Config, FeaturesConfig
from specialist_router.router.features import EntityVocab, Featurizer

FEATURES = FeaturesConfig(
    embedding="hashing",
    embed_dim=16,
    ngram_min=3,
    ngram_max=4,
    len_norm_chars=400.0,
    len_norm_tokens=80.0,
    count_norm=8.0,
)


def _featurizer(env_config: Config) -> Featurizer:
    return Featurizer(FEATURES, EntityVocab.from_db_config(env_config.db))


def test_feature_dim_and_names_align(env_config: Config) -> None:
    feat = _featurizer(env_config)
    vec = feat.transform("How much revenue did enterprise customers make in 2023?")
    assert vec.shape == (feat.feature_dim,)
    assert len(feat.feature_names) == feat.feature_dim
    assert feat.feature_names[0] == "bias"
    assert vec[0] == 1.0  # bias term


def test_deterministic(env_config: Config) -> None:
    feat = _featurizer(env_config)
    q = "List the top 3 product categories by units sold during 2022."
    assert np.array_equal(feat.transform(q), feat.transform(q))


def test_date_and_ranking_signals(env_config: Config) -> None:
    feat = _featurizer(env_config)
    names = feat.feature_names
    vec = feat.transform("List the top 5 categories by units sold in 2022.")
    assert vec[names.index("has_date_window")] == 1.0
    assert vec[names.index("has_topk_marker")] == 1.0
    assert vec[names.index("bucket_ranking")] == 1.0
    # No date, no ranking:
    plain = feat.transform("What is the answer?")
    assert plain[names.index("has_date_window")] == 0.0
    assert plain[names.index("bucket_aggregation")] == 1.0


def test_entities_counted(env_config: Config) -> None:
    feat = _featurizer(env_config)
    names = feat.feature_names
    with_entities = feat.transform("refund rate for enterprise customers on orders in beauty")
    none = feat.transform("what is the value")
    assert with_entities[names.index("n_entities")] > none[names.index("n_entities")]


def test_hashing_embedding_is_unit_norm(env_config: Config) -> None:
    feat = _featurizer(env_config)
    vec = feat.transform("revenue by segment during 2024")
    embed = vec[len(feat.feature_names) - FEATURES.embed_dim :]
    assert np.linalg.norm(embed) == pytest.approx(1.0)


def test_difficulty_tag_not_in_features(env_config: Config) -> None:
    feat = _featurizer(env_config)
    # The featurizer only sees the question string; there is no channel for the difficulty tag.
    assert "difficulty" not in " ".join(feat.feature_names)
