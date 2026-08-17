"""Property-based invariants for the verifier (hypothesis)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from specialist_router.config import VerifierConfig
from specialist_router.env.records import AnswerType
from specialist_router.env.verifier import verify

VCFG = VerifierConfig(
    money_abs_usd=0.01, ratio_rel=0.001, ratio_abs=1e-6, pp_rel=0.001, pp_abs=1e-4
)


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
def test_identical_scalar_is_always_correct(x: float) -> None:
    assert verify(x, x, AnswerType.RATIO, VCFG).correct
    assert verify(x, x, AnswerType.MONEY_USD, VCFG).correct
    assert verify(x, x, AnswerType.PERCENTAGE_POINTS, VCFG).correct


@given(
    e=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
    d=st.floats(min_value=-9e-7, max_value=9e-7),
)
def test_within_absolute_tolerance_is_correct(e: float, d: float) -> None:
    # ratio_abs is 1e-6, so any perturbation under it is within tolerance regardless of scale.
    assert verify(e, e + d, AnswerType.RATIO, VCFG).correct


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=100.0))
def test_far_beyond_tolerance_is_incorrect(e: float) -> None:
    # threshold = max(1e-6, 1e-3*|e|) <= 0.1 here, so an offset of 1.0 must be judged wrong.
    assert not verify(e, e + 1.0, AnswerType.RATIO, VCFG).correct


@given(st.integers(min_value=-10_000, max_value=10_000))
def test_integer_identity_is_correct(n: int) -> None:
    assert verify(n, n, AnswerType.INTEGER, VCFG).correct
    assert not verify(n, n + 1, AnswerType.INTEGER, VCFG).correct
