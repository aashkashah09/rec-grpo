"""Versioned, JSONL-serializable record schemas for the environment.

Centralising the record types (and a single ``SCHEMA_VERSION``) here keeps every artifact
we write — task datasets, trajectories, verdicts — on one explicitly versioned schema, so a
reviewer reading a committed JSONL always knows exactly how to interpret it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
"""Bump when any record shape below changes in a backwards-incompatible way."""

Difficulty = Literal["easy", "med", "hard"]

# The canonical answer value is deliberately a small closed union: a money/ratio/pp scalar
# (float), an id/count (int), or an ordered list of category names (list[str]). Keeping it
# closed lets the verifier dispatch on ``answer_type`` without ever parsing prose.
AnswerValue = float | int | list[str]


class AnswerType(StrEnum):
    """How an answer value is interpreted and tolerated by the verifier.

    The string values double as the stable on-disk tags in JSONL artifacts.
    """

    MONEY_USD = "money_usd"
    """A monetary amount in US dollars (compared with a one-cent absolute tolerance)."""

    RATIO = "ratio"
    """A dimensionless decimal fraction, possibly signed (e.g. a growth rate of 0.123)."""

    PERCENTAGE_POINTS = "percentage_points"
    """A signed difference of two rates expressed in percentage points (2.5 == 2.5 pp)."""

    INTEGER = "integer"
    """An exact integer such as a count or a primary-key id."""

    LIST_STR = "list_str"
    """An ordered list of strings (e.g. category names); order and length are significant."""


class Task(BaseModel):
    """A single procedurally-generated task with its programmatic ground truth.

    ``expected`` is computed by the pure-Python reference in ``env.tasks``; ``reference_sql``
    is the equivalent query the oracle agent runs through the sandbox. The two must agree.
    """

    schema_version: int = Field(default=SCHEMA_VERSION)
    task_id: str
    template_id: str
    difficulty: Difficulty
    question: str
    answer_type: AnswerType
    params: dict[str, object]
    expected: AnswerValue
    reference_sql: str


class ToolCall(BaseModel):
    """One tool invocation within an episode and its (stringified) result."""

    tool: str
    arguments: dict[str, object]
    result: str
    ok: bool


class Trajectory(BaseModel):
    """The structured record of one agent's attempt at one task."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    task_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: AnswerValue | None = None
    stop_reason: Literal["final_answer", "max_turns", "tool_budget", "error"]
    num_turns: int


class Verdict(BaseModel):
    """The verifier's typed judgement of a submitted answer against ground truth."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    correct: bool
    reason: str
    answer_type: AnswerType
    extracted: AnswerValue | None
    expected: AnswerValue
    tolerance: dict[str, float]
