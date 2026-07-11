"""A tool-using chat agent that drives the episode loop via JSON actions (Phase 4; not in CI).

This is the shared implementation behind both live arms: it turns the episode :class:`Observation`
into chat messages, asks a model for the next action as JSON, and parses that into the episode
loop's :class:`Action` type. It is provider-agnostic — the concrete ``api_agent`` and
``local_agent`` just bind it to different endpoints (ADR-005). No network here: the completion
function is injected, so this module has no heavy dependencies and stays importable in CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from specialist_router.agents.base import Action, FinalAnswer, Observation, ToolRequest
from specialist_router.agents.metering import Usage
from specialist_router.env.episode import system_prompt
from specialist_router.env.records import AnswerType, AnswerValue
from specialist_router.serving.clients import ChatResponse

CompleteFn = Callable[[list[dict[str, str]], int], ChatResponse]

_ACTION_INSTRUCTIONS = (
    "Respond with exactly one JSON object and nothing else. To call a tool: "
    '{"tool": "run_sql", "arguments": {"query": "SELECT ..."}} '
    "(tools: inspect_schema, run_sql, python_calc). To answer: "
    '{"final_answer": <value matching the task\'s answer format>}.'
)


def _fallback_answer(answer_type: AnswerType) -> AnswerValue:
    """A shape-valid but wrong answer, used when the model output cannot be parsed."""
    if answer_type is AnswerType.LIST_STR:
        return ["__unparseable__"]
    if answer_type is AnswerType.INTEGER:
        return -1
    return -999999.0


def _coerce_answer(value: object, answer_type: AnswerType) -> AnswerValue:
    """Coerce a parsed JSON value to an :class:`AnswerValue` (fallback if it is the wrong shape)."""
    if answer_type is AnswerType.LIST_STR:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return [str(v) for v in value]
        return _fallback_answer(answer_type)
    if isinstance(value, bool):  # bool is an int subclass but never a valid numeric answer
        return _fallback_answer(answer_type)
    if isinstance(value, (int, float)):
        return value
    return _fallback_answer(answer_type)


def _parse_action(text: str, answer_type: AnswerType) -> Action:
    """Parse the first JSON object in ``text`` into an :class:`Action` (fallback on failure)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return FinalAnswer(_fallback_answer(answer_type))
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return FinalAnswer(_fallback_answer(answer_type))
    if not isinstance(obj, dict):
        return FinalAnswer(_fallback_answer(answer_type))
    if "final_answer" in obj:
        return FinalAnswer(_coerce_answer(obj["final_answer"], answer_type))
    tool = obj.get("tool")
    if isinstance(tool, str):
        arguments = obj.get("arguments", {})
        return ToolRequest(tool, arguments if isinstance(arguments, dict) else {})
    return FinalAnswer(_fallback_answer(answer_type))


class ChatToolAgent:
    """An :class:`~specialist_router.agents.base.Agent` backed by an injected chat completion fn."""

    def __init__(self, name: str, complete_fn: CompleteFn, max_tokens: int = 512) -> None:
        """Create a fresh agent (one instance per episode — it holds conversation state)."""
        self.name = name
        self.usage = Usage()
        self._complete = complete_fn
        self._max_tokens = max_tokens
        self._messages: list[dict[str, str]] | None = None

    def act(self, obs: Observation) -> Action:
        """Build the next prompt, request a completion, and parse it into an action."""
        if self._messages is None:
            self._messages = [
                {"role": "system", "content": system_prompt(obs.task, obs.schema)},
                {"role": "user", "content": _ACTION_INSTRUCTIONS},
            ]
        elif obs.last is not None:
            self._messages.append(
                {
                    "role": "user",
                    "content": f"Tool {obs.last.tool} result:\n{obs.last.text}\n"
                    "Respond with the next action as JSON.",
                }
            )
        response = self._complete(self._messages, self._max_tokens)
        self.usage.add(response.prompt_tokens, response.completion_tokens)
        self._messages.append({"role": "assistant", "content": response.text})
        return _parse_action(response.text, obs.task.answer_type)
