"""Versioned, JSONL-serializable record schemas for the environment.

Centralising the record types (and a single ``SCHEMA_VERSION``) here keeps every artifact
we write — task datasets, trajectories, verdicts — on one explicitly versioned schema, so a
reviewer reading a committed JSONL always knows exactly how to interpret it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2
"""Bump when any record shape below changes in a backwards-incompatible way.

Version history:
* 1 — Phase 1: ``Task``, ``Trajectory``, ``ToolCall``, ``Verdict``.
* 2 — Phase 2: adds :class:`RouterDecision` (propensity-logged routing decisions). Existing
  Phase-1 records are unchanged in shape; committed v1 artifacts remain readable because every
  record self-describes its ``schema_version``.
"""

Difficulty = Literal["easy", "med", "hard"]

# The router is a two-arm contextual bandit: the cheap local specialist vs the frontier API.
# Keeping the arm set a closed literal lets policies, the logger, and OPE dispatch exhaustively.
Arm = Literal["local", "api"]
ARMS: tuple[Arm, Arm] = ("local", "api")

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


class RouterDecision(BaseModel):
    """One propensity-logged routing decision — the single unit OPE evaluates.

    This is a *contextual bandit* record: one context (``feature_vector``), one action
    (``action``), one scalar ``reward``. It deliberately carries every quantity an off-policy
    estimator or a replay needs so the estimators are pure functions of a list of these records
    and results reproduce from the committed JSONL alone:

    * the context and the exact featurizer output width (``feature_dim``/``feature_names``),
    * the acting policy's identity and a hash of its parameters (audit / grouping),
    * the logged action and its **propensity under the logging policy** ``π₀(action | x)`` — the
      denominator of every importance weight, so it is constrained ``> 0``,
    * the decomposed reward (``quality``, ``cost_usd``, ``latency_s`` and their normalized forms)
      **and the reward weights + reference scales** (``reward_lambda``, ``reward_mu``,
      ``cost_ref_usd``, ``latency_ref_s``) so the composite reward is recomputable and a replay
      can assert it is on the same scale as the OPE it validates.

    Scope note (``CLAUDE.md`` rule #2): this record is the routing decision only. There is no
    per-turn or trajectory-level importance weighting of the agent's tool-use rollout anywhere.
    """

    schema_version: int = Field(default=SCHEMA_VERSION)
    decision_id: str
    task_id: str
    template_id: str
    difficulty: Difficulty
    """The true difficulty tag — logged for post-hoc analysis only; never fed to the policy."""

    feature_names: list[str]
    feature_vector: list[float]
    feature_dim: int

    policy_name: str
    policy_version: str
    policy_params_hash: str

    action: Arm
    propensity: float = Field(gt=0.0, le=1.0)
    """``π₀(action | x)`` under the logging policy; strictly positive so IPS weights are finite."""

    all_propensities: dict[str, float]

    quality: int = Field(ge=0, le=1)
    cost_usd: float = Field(ge=0.0)
    latency_s: float = Field(ge=0.0)
    cost_norm: float = Field(ge=0.0, le=1.0)
    latency_norm: float = Field(ge=0.0, le=1.0)
    reward: float

    reward_lambda: float = Field(ge=0.0)
    reward_mu: float = Field(ge=0.0)
    cost_ref_usd: float = Field(gt=0.0)
    latency_ref_s: float = Field(gt=0.0)

    agent_versions: dict[str, str]
    seed: int
    timestamp: str
