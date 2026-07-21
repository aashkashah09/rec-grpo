import numpy as np

from recgrpo.data.dataset import EvalExample
from recgrpo.eval.bootstrap import bootstrap_difference
from recgrpo.eval.slate_analysis import composition, per_example_cold_positions


def setup():
    cold = np.zeros(10, dtype=bool)
    cold[[6, 7, 8, 9]] = True
    visible = np.zeros(10, dtype=bool)
    visible[[6, 7]] = True  # the reward only ever scored two of the four
    examples = [
        EvalExample(user=0, history=[], target=6, timestamp=1),
        EvalExample(user=1, history=[], target=0, timestamp=2),
    ]
    return cold, visible, examples


def test_composition_counts_positions_and_conversions():
    cold, visible, examples = setup()
    slates = [[6, 8, 0, 1, 2], [9, 1, 2, 3, 4]]
    row = composition("test", slates, examples, cold, visible, k=5)

    assert row.n_positions == 10
    assert row.cold_positions == 3
    assert row.reward_visible_positions == 1
    assert row.never_rewarded_positions == 2
    assert row.cold_hits == 1
    assert row.cold_conversion == 1 / 3


def test_never_rewarded_share_is_measured_separately():
    cold, visible, examples = setup()
    slates = [[8, 9, 0, 1, 2], [6, 1, 2, 3, 4]]
    counts = per_example_cold_positions(slates, cold, visible, k=5, subset="never_rewarded")
    assert counts.tolist() == [0.4, 0.0]


def test_bootstrap_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(60), 5)
    a = rng.normal(0.01, 0.005, size=users.shape)
    b = a + rng.normal(0.005, 0.002, size=users.shape)

    stats = bootstrap_difference(a, b, users, n_replicates=500, seed=1)
    assert stats["ci_low"] < stats["observed"] < stats["ci_high"]
    assert stats["n_users"] == 60
    assert stats["p_value"] < 0.05
