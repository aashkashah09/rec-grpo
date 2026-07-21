from math import log2

import numpy as np

from recgrpo.eval.metrics import RankingMetrics, ndcg_at_k, recall_at_k
from recgrpo.reward import SlateReward, group_advantages, ndcg


def cold_mask(n=20, cold=(15, 16, 17, 18, 19)) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[list(cold)] = True
    return mask


def test_ndcg_matches_the_hand_computation():
    slate = [4, 1, 7, 2]
    targets = {1, 2}
    expected = (1 / log2(3) + 1 / log2(5)) / (1 / log2(2) + 1 / log2(3))
    assert ndcg(slate, targets, k=10) == expected


def test_single_target_metrics():
    slate = [9, 3, 5]
    assert recall_at_k(slate, 3, k=2) == 1.0
    assert recall_at_k(slate, 5, k=2) == 0.0
    assert ndcg_at_k(slate, 3, k=3) == 1 / log2(3)
    assert ndcg_at_k(slate, 5, k=2) == 0.0


def test_cold_bonus_is_capped():
    reward = SlateReward(cold_mask(), cold_lambda=1.0, cold_cap=2)
    targets = {15, 16, 17, 18}

    one = reward([15, 0, 1, 2, 3, 4, 5, 6, 7, 8], targets)
    two = reward([15, 16, 1, 2, 3, 4, 5, 6, 7, 8], targets)
    four = reward([15, 16, 17, 18, 3, 4, 5, 6, 7, 8], targets)

    assert one.cold_bonus == 0.5
    assert two.cold_bonus == 1.0
    assert four.cold_bonus == 1.0
    assert four.cold_hits == 4


def test_uncapped_bonus_keeps_paying():
    slate, targets = [15, 16, 17, 18], {15, 16, 17, 18}
    capped = SlateReward(cold_mask(), cold_lambda=1.0, cold_cap=2)(slate, targets)
    uncapped = SlateReward(cold_mask(), cold_lambda=1.0, cold_cap=0)(slate, targets)

    assert capped.cold_bonus == 1.0          # tops out at the cap
    assert uncapped.cold_bonus == 4.0        # every hit still pays
    assert uncapped.total > capped.total
    # with no cap the bonus alone outweighs any achievable ranking score
    assert uncapped.cold_bonus > 1.0 >= uncapped.ndcg


def test_cap_size_sets_the_per_hit_rate():
    targets = {15, 16, 17, 18}
    one_hit = [15, 0, 1]
    assert SlateReward(cold_mask(), cold_cap=1)(one_hit, targets).cold_bonus == 1.0
    assert SlateReward(cold_mask(), cold_cap=2)(one_hit, targets).cold_bonus == 0.5
    assert SlateReward(cold_mask(), cold_cap=5)(one_hit, targets).cold_bonus == 0.2


def test_only_cold_hits_pay_the_bonus():
    reward = SlateReward(cold_mask(), cold_lambda=1.0, cold_cap=2)
    # cold items the user never engaged with earn nothing
    assert reward([15, 16, 17], targets=set()).cold_bonus == 0.0
    # warm hits earn relevance but no bonus
    warm = reward([3], targets={3})
    assert warm.cold_bonus == 0.0
    assert warm.ndcg > 0


def test_lambda_scales_only_the_bonus():
    slate, targets = [15, 3], {15, 3}
    zero = SlateReward(cold_mask(), cold_lambda=0.0)(slate, targets)
    four = SlateReward(cold_mask(), cold_lambda=4.0)(slate, targets)
    assert zero.ndcg == four.ndcg
    assert four.total - zero.total == 4.0 * four.cold_bonus


def test_group_advantages_are_zero_mean_and_flat_groups_contribute_nothing():
    rewards = np.array([0.1, 0.4, 0.2, 0.3])
    adv = group_advantages(rewards)
    assert abs(adv.mean()) < 1e-9
    assert adv.argmax() == rewards.argmax()

    assert np.allclose(group_advantages(np.full(4, 0.7)), 0.0)


def test_metrics_slices_share_the_same_slates():
    metrics = RankingMetrics(ks=(1, 2))
    metrics.update([5, 9], target=9, slices=("cold",))
    metrics.update([1, 2], target=7)
    result = metrics.result()

    assert result["all"]["n"] == 2
    assert result["all"]["recall@2"] == 0.5
    assert result["cold"]["n"] == 1
    assert result["cold"]["recall@2"] == 1.0
    assert result["cold"]["recall@1"] == 0.0
