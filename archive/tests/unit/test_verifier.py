"""Verifier edge cases: units, rounding, tolerances, list order, and malformed answers."""

from __future__ import annotations

import pytest

from specialist_router.config import Config, VerifierConfig
from specialist_router.env.records import AnswerType
from specialist_router.env.verifier import verify


@pytest.fixture
def vcfg(env_config: Config) -> VerifierConfig:
    return env_config.verifier


def test_money_within_one_cent_is_correct(vcfg: VerifierConfig) -> None:
    assert verify(100.00, 100.004, AnswerType.MONEY_USD, vcfg).correct
    assert verify(100.00, 99.995, AnswerType.MONEY_USD, vcfg).correct


def test_money_off_by_more_than_a_cent_is_wrong(vcfg: VerifierConfig) -> None:
    assert not verify(100.00, 100.02, AnswerType.MONEY_USD, vcfg).correct


def test_money_dollars_vs_cents_confusion_is_wrong(vcfg: VerifierConfig) -> None:
    # Answering in cents (12345) when dollars (123.45) were expected must fail.
    assert not verify(123.45, 12345.0, AnswerType.MONEY_USD, vcfg).correct


def test_ratio_fraction_vs_percentage_confusion_is_wrong(vcfg: VerifierConfig) -> None:
    # Expected fraction 0.12; answering 12 (as if a percent) must fail.
    assert not verify(0.12, 12.0, AnswerType.RATIO, vcfg).correct
    assert verify(0.12, 0.1200005, AnswerType.RATIO, vcfg).correct


def test_percentage_points_tolerance(vcfg: VerifierConfig) -> None:
    assert verify(2.5, 2.5004, AnswerType.PERCENTAGE_POINTS, vcfg).correct
    assert not verify(2.5, 3.0, AnswerType.PERCENTAGE_POINTS, vcfg).correct


def test_integer_accepts_float_valued_integer(vcfg: VerifierConfig) -> None:
    assert verify(5, 5.0, AnswerType.INTEGER, vcfg).correct
    assert not verify(5, 5.5, AnswerType.INTEGER, vcfg).correct
    assert not verify(5, 6, AnswerType.INTEGER, vcfg).correct


def test_list_order_and_case(vcfg: VerifierConfig) -> None:
    assert verify(
        ["electronics", "apparel"], ["Electronics", " apparel "], AnswerType.LIST_STR, vcfg
    ).correct
    # Wrong order must fail.
    assert not verify(
        ["electronics", "apparel"], ["apparel", "electronics"], AnswerType.LIST_STR, vcfg
    ).correct
    # Wrong length must fail.
    assert not verify(
        ["electronics", "apparel"], ["electronics"], AnswerType.LIST_STR, vcfg
    ).correct


def test_none_answer_is_incorrect_not_error(vcfg: VerifierConfig) -> None:
    verdict = verify(100.0, None, AnswerType.MONEY_USD, vcfg)
    assert not verdict.correct
    assert verdict.extracted is None


def test_wrong_shape_answer_is_incorrect(vcfg: VerifierConfig) -> None:
    # A list where a number is expected -> incorrect, not a crash.
    assert not verify(1.0, ["x"], AnswerType.RATIO, vcfg).correct
    # A number where a list is expected -> incorrect.
    assert not verify(["a"], 1.0, AnswerType.LIST_STR, vcfg).correct


def test_bool_is_never_a_valid_number(vcfg: VerifierConfig) -> None:
    assert not verify(1.0, True, AnswerType.RATIO, vcfg).correct


def test_malformed_expected_raises(vcfg: VerifierConfig) -> None:
    with pytest.raises(ValueError):
        verify(["a"], 1.0, AnswerType.MONEY_USD, vcfg)
    with pytest.raises(ValueError):
        verify(1.5, 1, AnswerType.INTEGER, vcfg)
