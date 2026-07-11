"""Property-based invariants for the python_calc AST-whitelist evaluator (hypothesis)."""

from __future__ import annotations

import operator
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from specialist_router.config import PythonCalcConfig
from specialist_router.env.tools import CalcError, python_calc

CFG = PythonCalcConfig(max_nodes=200, max_exponent=8)

_OPS: dict[str, Callable[[int, int], int]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


@given(
    a=st.integers(min_value=-1000, max_value=1000),
    b=st.integers(min_value=-1000, max_value=1000),
    op=st.sampled_from(list(_OPS)),
)
def test_arithmetic_matches_python(a: int, b: int, op: str) -> None:
    assert python_calc(f"({a}) {op} ({b})", CFG) == float(_OPS[op](a, b))


@given(
    a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=1e6),
)
def test_division_matches_python(a: float, b: float) -> None:
    assert python_calc(f"({a}) / ({b})", CFG) == pytest.approx(a / b)


@given(st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True))
def test_bare_names_are_always_rejected(name: str) -> None:
    # Any lone identifier is a Name/Constant node with no numeric value -> rejected.
    with pytest.raises(CalcError):
        python_calc(name, CFG)
