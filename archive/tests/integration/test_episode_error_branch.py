"""The deferred Phase-1 coverage: the episode loop's ``error`` stop-reason branch.

When a (real) agent raises mid-episode — a transport error, a timeout, an unparseable response —
``run_episode`` must record a structured ``error`` outcome rather than propagate the exception and
abort a whole traffic run. Phase 1 added the branch's target ``stop_reason`` to the schema but had
no agent that could raise; this test closes that gap.
"""

from __future__ import annotations

from pathlib import Path

from specialist_router.agents.base import Action, Observation
from specialist_router.config import Config
from specialist_router.env.database import Dataset
from specialist_router.env.episode import run_episode
from specialist_router.env.tasks import generate_tasks
from specialist_router.env.tools import ToolContext


class _RaisingAgent:
    """An agent that always raises when asked to act."""

    name = "raiser"

    def act(self, obs: Observation) -> Action:
        raise RuntimeError("boom")


class _RaisingOnSecondTurn:
    """Answers with a tool call once, then raises — exercising the branch mid-episode."""

    name = "raise-later"

    def __init__(self) -> None:
        self._calls = 0

    def act(self, obs: Observation) -> Action:
        from specialist_router.agents.base import ToolRequest

        self._calls += 1
        if self._calls == 1:
            return ToolRequest("inspect_schema", {})
        raise RuntimeError("boom on turn 2")


def test_agent_raising_on_first_turn_yields_error(
    env_config: Config, mini_dataset: Dataset, mini_db_file: Path
) -> None:
    task = generate_tasks(mini_dataset, env_config)[0]
    tools = ToolContext(mini_db_file, env_config.tools.run_sql, env_config.tools.python_calc)
    traj = run_episode(task, _RaisingAgent(), tools, env_config.episode)
    assert traj.stop_reason == "error"
    assert traj.num_turns == 1
    assert traj.final_answer is None
    assert traj.tool_calls == []
    tools.close()


def test_agent_raising_mid_episode_yields_error_with_partial_calls(
    env_config: Config, mini_dataset: Dataset, mini_db_file: Path
) -> None:
    task = generate_tasks(mini_dataset, env_config)[0]
    tools = ToolContext(mini_db_file, env_config.tools.run_sql, env_config.tools.python_calc)
    traj = run_episode(task, _RaisingOnSecondTurn(), tools, env_config.episode)
    assert traj.stop_reason == "error"
    assert traj.num_turns == 2
    assert len(traj.tool_calls) == 1  # the first-turn inspect_schema was recorded
    tools.close()
