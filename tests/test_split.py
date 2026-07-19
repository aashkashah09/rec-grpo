import numpy as np
import pandas as pd
import pytest

from recgrpo.data.dataset import build_eval_examples
from recgrpo.data.split import cold_items, cold_target_count, temporal_split


@pytest.fixture
def interactions() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 1000
    return pd.DataFrame(
        {
            "user": rng.integers(0, 50, n),
            "item": rng.integers(0, 80, n),
            "timestamp": np.sort(rng.integers(1_000_000, 2_000_000, n)),
        }
    )


def test_block_sizes_follow_the_requested_fractions(interactions):
    split = temporal_split(interactions, val_frac=0.1, test_frac=0.1)
    assert split.sizes == {"train": 800, "val": 100, "test": 100}
    assert sum(split.sizes.values()) == len(interactions)


def test_blocks_do_not_overlap_in_time(interactions):
    split = temporal_split(interactions)
    assert split.train["timestamp"].max() <= split.val["timestamp"].min()
    assert split.val["timestamp"].max() <= split.test["timestamp"].min()


def test_cold_items_have_no_training_interaction(interactions):
    split = temporal_split(interactions)
    cold = cold_items(split, n_items=80)
    assert not set(np.flatnonzero(cold)) & set(split.train["item"])
    assert cold_target_count(split, cold) == int(cold[split.test["item"].to_numpy()].sum())


def test_every_target_is_evaluated(interactions):
    split = temporal_split(interactions)
    examples = build_eval_examples(interactions, split.test, max_history_items=50)
    assert len(examples) == len(split.test)
    assert [e.target for e in examples] == split.test["item"].tolist()


def test_prompts_only_contain_the_past(interactions):
    split = temporal_split(interactions)
    examples = build_eval_examples(interactions, split.test, max_history_items=50)
    lookup = interactions.set_index(["user", "item"])["timestamp"].groupby(level=[0, 1]).min()
    for example in examples:
        for item in example.history:
            assert lookup.loc[(example.user, item)] < example.timestamp
