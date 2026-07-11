"""Unit tests for the propensity-log I/O, the logged-dataset view, and the record schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from specialist_router.env.records import SCHEMA_VERSION, RouterDecision
from specialist_router.router.logger import (
    DecisionLogger,
    LoggedDataset,
    LogIntegrityError,
    read_decisions,
)
from tests.support import make_decision


def test_router_decision_is_schema_v2() -> None:
    assert SCHEMA_VERSION == 2
    assert make_decision().schema_version == 2


def test_propensity_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        make_decision(propensity=0.0)
    with pytest.raises(ValidationError):
        make_decision(propensity=-0.1)


def test_quality_must_be_binary() -> None:
    with pytest.raises(ValidationError):
        make_decision(quality=2)


def test_write_read_round_trip(tmp_path: Path) -> None:
    decisions = [make_decision(decision_id=f"dec-{i}") for i in range(5)]
    path = tmp_path / "log.jsonl"
    with DecisionLogger(path) as logger:
        for d in decisions:
            logger.write(d)
    read_back = read_decisions(path)
    assert [d.decision_id for d in read_back] == [d.decision_id for d in decisions]


def test_logger_requires_context_manager(tmp_path: Path) -> None:
    logger = DecisionLogger(tmp_path / "log.jsonl")
    with pytest.raises(RuntimeError, match="context manager"):
        logger.write(make_decision())


def test_read_rejects_nonpositive_propensity(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    # Hand-forge a line with propensity 0 (bypassing the constructor) to prove read-side validation.
    good = make_decision().model_dump()
    good["propensity"] = 0.0
    path.write_text(RouterDecision.model_construct(**good).model_dump_json() + "\n")
    with pytest.raises((ValidationError, LogIntegrityError)):
        read_decisions(path)


def test_logged_dataset_alignment() -> None:
    decisions = [
        make_decision(decision_id="a", action="local", reward=0.9, feature_vector=[1.0, 0.2]),
        make_decision(decision_id="b", action="api", reward=0.1, feature_vector=[1.0, 0.8]),
    ]
    data = LoggedDataset.from_decisions(decisions)
    assert data.n == 2
    assert data.d == 2
    assert data.arm_mask("local").tolist() == [True, False]
    assert np.allclose(data.reward, [0.9, 0.1])


def test_logged_dataset_rejects_width_mismatch() -> None:
    decisions = [
        make_decision(decision_id="a", feature_vector=[1.0, 0.2]),
        make_decision(decision_id="b", feature_vector=[1.0, 0.2, 0.3]),
    ]
    with pytest.raises(LogIntegrityError, match="feature width"):
        LoggedDataset.from_decisions(decisions)


def test_logged_dataset_rejects_empty() -> None:
    with pytest.raises(LogIntegrityError, match="zero decisions"):
        LoggedDataset.from_decisions([])
