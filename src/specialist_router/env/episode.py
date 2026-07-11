"""The multi-turn episode loop: drive an agent over the tools under strict budgets.

The loop owns the turn count and tool budget (``CLAUDE.md``: max 12 turns, tool budget) so no
agent can exceed them, dispatches tool requests to the sandbox catching every sandbox error,
and records a structured :class:`Trajectory`. Verification is deliberately a separate step
(:func:`run_and_verify`) so the trust-critical verifier stays a pure function of
``(expected, submitted)``.
"""

from __future__ import annotations

from specialist_router.agents.base import (
    Action,
    Agent,
    FinalAnswer,
    Observation,
    ToolOutcome,
    ToolRequest,
)
from specialist_router.config import EpisodeConfig, VerifierConfig
from specialist_router.env.records import Task, ToolCall, Trajectory, Verdict
from specialist_router.env.tools import SandboxError, ToolContext
from specialist_router.env.verifier import verify

_TOOL_SCHEMA = """\
You have three tools. Respond with exactly one action per turn.
- inspect_schema(): returns tables, column types, and foreign keys.
- run_sql(query): runs one read-only SELECT/WITH statement and returns rows.
- python_calc(expr): evaluates a numeric expression (arithmetic + abs/round/min/max/sqrt).
Submit your result with final_answer(value), where value matches the task's answer format.\
"""


def system_prompt(task: Task, schema: str) -> str:
    """Build the system prompt embedding the tool schema and the database schema.

    Scripted reference agents ignore this; it exists so the Phase 2 model agents share one
    canonical prompt with the same tool contract.
    """
    return (
        "You are a careful SQL analytics agent.\n"
        f"{_TOOL_SCHEMA}\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Task: {task.question}"
    )


def _dispatch(request: ToolRequest, tools: ToolContext) -> ToolOutcome:
    """Execute one tool request, converting any sandbox rejection into a failed outcome."""
    try:
        if request.tool == "inspect_schema":
            return ToolOutcome("inspect_schema", True, tools.inspect_schema())
        if request.tool == "run_sql":
            result = tools.run_sql(str(request.arguments["query"]))
            return ToolOutcome("run_sql", True, result.as_text(), sql_result=result)
        if request.tool == "python_calc":
            value = tools.python_calc(str(request.arguments["expr"]))
            return ToolOutcome("python_calc", True, str(value))
        return ToolOutcome(
            request.tool, False, f"unknown tool: {request.tool}", error="unknown_tool"
        )
    except SandboxError as exc:
        return ToolOutcome(request.tool, False, f"error: {exc}", error=str(exc))
    except KeyError as exc:
        return ToolOutcome(
            request.tool, False, f"missing argument: {exc}", error="missing_argument"
        )


def run_episode(task: Task, agent: Agent, tools: ToolContext, config: EpisodeConfig) -> Trajectory:
    """Run ``agent`` on ``task`` and return its :class:`Trajectory`.

    Args:
        task: The task to solve.
        agent: A per-episode agent implementing the :class:`Agent` protocol.
        tools: The sandboxed tool context bound to the task's database.
        config: Turn and tool-call budgets.

    Returns:
        The trajectory, whose ``stop_reason`` records why the episode ended.
    """
    schema = tools.inspect_schema()
    obs = Observation(task=task, schema=schema, last=None)
    tool_calls: list[ToolCall] = []

    for turn in range(1, config.max_turns + 1):
        try:
            action: Action = agent.act(obs)
        except Exception:
            # A real (vLLM/API) agent can fail mid-episode — a transport error, a timeout, an
            # unparseable response that raises. The loop owns robustness, so we record a
            # structured ``error`` outcome (never silently) rather than letting it propagate and
            # abort a whole traffic run. The trust-critical verifier stays a separate, pure step.
            return Trajectory(
                task_id=task.task_id,
                tool_calls=tool_calls,
                final_answer=None,
                stop_reason="error",
                num_turns=turn,
            )
        if isinstance(action, FinalAnswer):
            return Trajectory(
                task_id=task.task_id,
                tool_calls=tool_calls,
                final_answer=action.value,
                stop_reason="final_answer",
                num_turns=turn,
            )
        if len(tool_calls) >= config.tool_budget:
            return Trajectory(
                task_id=task.task_id,
                tool_calls=tool_calls,
                final_answer=None,
                stop_reason="tool_budget",
                num_turns=turn,
            )
        outcome = _dispatch(action, tools)
        tool_calls.append(
            ToolCall(
                tool=action.tool, arguments=action.arguments, result=outcome.text, ok=outcome.ok
            )
        )
        obs = Observation(task=task, schema=schema, last=outcome)

    return Trajectory(
        task_id=task.task_id,
        tool_calls=tool_calls,
        final_answer=None,
        stop_reason="max_turns",
        num_turns=config.max_turns,
    )


def run_and_verify(
    task: Task,
    agent: Agent,
    tools: ToolContext,
    episode_config: EpisodeConfig,
    verifier_config: VerifierConfig,
) -> tuple[Trajectory, Verdict]:
    """Run an episode and verify its final answer against the task's ground truth."""
    trajectory = run_episode(task, agent, tools, episode_config)
    verdict = verify(task.expected, trajectory.final_answer, task.answer_type, verifier_config)
    return trajectory, verdict
