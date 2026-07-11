"""The agent protocol and the action/observation types the episode loop exchanges.

An agent is a stateful object that, given the current :class:`Observation`, returns one
:class:`Action` per turn — either a tool request or a final answer. The episode loop (not the
agent) enforces turn and tool budgets, so a misbehaving agent cannot exceed them. This same
protocol will host the real vLLM/API agents in Phase 2; Phase 1 ships only deterministic
reference agents used to validate the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from specialist_router.env.records import AnswerValue, Task
from specialist_router.env.tools import SqlResult


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """An agent's request to invoke a tool with arguments."""

    tool: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """An agent's typed final answer, ending the episode."""

    value: AnswerValue


Action = ToolRequest | FinalAnswer


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of dispatching a :class:`ToolRequest`.

    ``sql_result`` carries the structured rows for ``run_sql`` so programmatic agents can read
    typed values; a model agent uses only ``text``.
    """

    tool: str
    ok: bool
    text: str
    sql_result: SqlResult | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """What an agent sees at the start of a turn."""

    task: Task
    schema: str
    last: ToolOutcome | None = field(default=None)


class Agent(Protocol):
    """A per-episode decision policy: map an observation to the next action."""

    name: str

    def act(self, obs: Observation) -> Action:
        """Return the next :class:`Action` given the current observation."""
        ...
