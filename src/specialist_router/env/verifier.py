"""Deterministic, value-based verification of submitted answers against ground truth.

The verifier never inspects model prose (``CLAUDE.md`` hard rule #3): it receives a typed
answer value (produced by the ``final_answer`` tool) and compares it to the programmatic
ground truth under a per-answer-type tolerance. A malformed *submitted* answer yields an
incorrect verdict (the agent's fault); a malformed *expected* value raises (our bug) — the
trust-critical path must not fail silently.
"""

from __future__ import annotations

from specialist_router.config import VerifierConfig
from specialist_router.env.records import AnswerType, AnswerValue, Verdict


def _as_number(value: AnswerValue | None) -> float | None:
    """Coerce a submitted value to float, or None if it is not a real number."""
    if isinstance(value, bool):  # bool is an int subclass; never a valid numeric answer
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: AnswerValue | None) -> int | None:
    """Coerce a submitted value to int when it is integral (accepts 5 or 5.0), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _require_number(value: AnswerValue, field: str) -> float:
    number = _as_number(value)
    if number is None:
        raise ValueError(f"expected {field} to be numeric, got {value!r}")
    return number


def verify(
    expected: AnswerValue,
    submitted: AnswerValue | None,
    answer_type: AnswerType,
    config: VerifierConfig,
) -> Verdict:
    """Compare ``submitted`` to ``expected`` under the tolerance for ``answer_type``.

    Args:
        expected: The programmatic ground truth (must match ``answer_type``; else raises).
        submitted: The agent's typed answer, or None if it never answered.
        answer_type: How to interpret and tolerance the comparison.
        config: Numeric tolerances.

    Returns:
        A :class:`Verdict` with ``correct``, a human-readable ``reason``, and the tolerance used.

    Raises:
        ValueError: If ``expected`` does not match ``answer_type`` (a generator bug).
    """
    if answer_type is AnswerType.LIST_STR:
        return _verify_list(expected, submitted)
    if answer_type is AnswerType.INTEGER:
        return _verify_integer(expected, submitted)
    if answer_type is AnswerType.MONEY_USD:
        return _verify_money(expected, submitted, config)
    if answer_type is AnswerType.RATIO:
        return _verify_scalar(
            expected, submitted, AnswerType.RATIO, config.ratio_abs, config.ratio_rel
        )
    if answer_type is AnswerType.PERCENTAGE_POINTS:
        return _verify_scalar(
            expected, submitted, AnswerType.PERCENTAGE_POINTS, config.pp_abs, config.pp_rel
        )
    raise ValueError(f"unhandled answer type: {answer_type}")  # pragma: no cover - exhaustive


def _verify_list(expected: AnswerValue, submitted: AnswerValue | None) -> Verdict:
    if not isinstance(expected, list):
        raise ValueError(f"expected a list for list_str, got {expected!r}")
    if not isinstance(submitted, list) or not all(isinstance(s, str) for s in submitted):
        return _verdict(
            False,
            "submitted answer is not a list of strings",
            AnswerType.LIST_STR,
            None,
            expected,
            {},
        )
    norm_expected = [s.strip().lower() for s in expected]
    norm_submitted = [s.strip().lower() for s in submitted]
    correct = norm_submitted == norm_expected
    reason = "ordered list matches" if correct else "ordered list differs (order/length/values)"
    return _verdict(correct, reason, AnswerType.LIST_STR, submitted, expected, {})


def _verify_integer(expected: AnswerValue, submitted: AnswerValue | None) -> Verdict:
    exp_int = _as_int(expected)
    if exp_int is None:
        raise ValueError(f"expected an integer, got {expected!r}")
    sub_int = _as_int(submitted)
    if sub_int is None:
        return _verdict(
            False,
            "submitted answer is not an integer",
            AnswerType.INTEGER,
            None,
            expected,
            {"abs": 0.0},
        )
    correct = sub_int == exp_int
    return _verdict(
        correct,
        "integer matches" if correct else "integer differs",
        AnswerType.INTEGER,
        sub_int,
        exp_int,
        {"abs": 0.0},
    )


def _verify_money(
    expected: AnswerValue, submitted: AnswerValue | None, config: VerifierConfig
) -> Verdict:
    exp = _require_number(expected, "expected money")
    tol = {"abs_usd": config.money_abs_usd}
    sub = _as_number(submitted)
    if sub is None:
        return _verdict(
            False, "submitted answer is not a number", AnswerType.MONEY_USD, None, expected, tol
        )
    correct = abs(round(sub, 2) - round(exp, 2)) <= config.money_abs_usd + 1e-9
    reason = "within one cent" if correct else f"off by ${abs(sub - exp):.4f}"
    return _verdict(correct, reason, AnswerType.MONEY_USD, sub, exp, tol)


def _verify_scalar(
    expected: AnswerValue,
    submitted: AnswerValue | None,
    answer_type: AnswerType,
    abs_tol: float,
    rel_tol: float,
) -> Verdict:
    exp = _require_number(expected, f"expected {answer_type}")
    tol = {"abs": abs_tol, "rel": rel_tol}
    sub = _as_number(submitted)
    if sub is None:
        return _verdict(False, "submitted answer is not a number", answer_type, None, expected, tol)
    threshold = max(abs_tol, rel_tol * abs(exp))
    correct = abs(sub - exp) <= threshold
    reason = "within tolerance" if correct else f"off by {abs(sub - exp):.6g} (tol {threshold:.6g})"
    return _verdict(correct, reason, answer_type, sub, exp, tol)


def _verdict(
    correct: bool,
    reason: str,
    answer_type: AnswerType,
    extracted: AnswerValue | None,
    expected: AnswerValue,
    tolerance: dict[str, float],
) -> Verdict:
    return Verdict(
        correct=correct,
        reason=reason,
        answer_type=answer_type,
        extracted=extracted,
        expected=expected,
        tolerance=tolerance,
    )
