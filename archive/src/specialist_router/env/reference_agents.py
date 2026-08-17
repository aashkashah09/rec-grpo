"""Deterministic reference agents used to validate the environment (no LLM involved).

* :class:`OracleAgent` runs each task's ``reference_sql`` through the sandbox and submits the
  parsed value — it must score 100% (proving tasks, tools, episode, and verifier all agree).
* :class:`TrivialAgent` submits a shape-valid but deliberately wrong answer — it must score
  ~0% (proving the verifier is not trivially satisfiable).

These pin down the Phase-1 done-condition. The real vLLM/API agents arrive in Phase 2.
"""

from __future__ import annotations

from specialist_router.agents.base import Action, FinalAnswer, Observation, ToolRequest
from specialist_router.env.records import AnswerType, AnswerValue
from specialist_router.env.tools import SqlResult


class OracleAgent:
    """Runs the task's reference SQL, then submits the parsed answer."""

    name = "oracle"

    def act(self, obs: Observation) -> Action:
        """First turn: request the reference SQL. Second turn: parse and submit the result."""
        if obs.last is None:
            return ToolRequest("run_sql", {"query": obs.task.reference_sql})
        if obs.last.sql_result is None:
            # The sandbox rejected the reference SQL — surface as no answer (fails loudly in tests).
            return FinalAnswer(_default_wrong(obs.task.answer_type))
        return FinalAnswer(_parse_sql(obs.last.sql_result, obs.task.answer_type))


def _parse_sql(result: SqlResult, answer_type: AnswerType) -> AnswerValue:
    """Convert a sandboxed SQL result into a typed answer value for ``answer_type``."""
    if answer_type is AnswerType.LIST_STR:
        return [str(row[0]) for row in result.rows]
    scalar = result.scalar()
    if answer_type is AnswerType.INTEGER:
        return int(scalar) if isinstance(scalar, (int, float)) else -1
    return float(scalar) if isinstance(scalar, (int, float)) else 0.0


class TrivialAgent:
    """Submits a well-formed but essentially-never-correct answer on the first turn."""

    name = "trivial"

    def act(self, obs: Observation) -> Action:
        """Immediately submit a deliberately wrong answer of the correct type."""
        return FinalAnswer(_default_wrong(obs.task.answer_type))


def _default_wrong(answer_type: AnswerType) -> AnswerValue:
    """Return a shape-valid sentinel that is wrong for essentially every task."""
    if answer_type is AnswerType.LIST_STR:
        return ["__none__"]
    if answer_type is AnswerType.INTEGER:
        return -1
    return -999999.0
