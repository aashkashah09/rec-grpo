"""End-to-end Phase-1 done-condition: oracle scores 100%, trivial ~0%, budgets are enforced."""

from __future__ import annotations

from pathlib import Path

from specialist_router.agents.base import Action, Agent, Observation, ToolRequest
from specialist_router.config import Config
from specialist_router.env.database import build_dataset, write_sqlite_file
from specialist_router.env.episode import run_and_verify, run_episode
from specialist_router.env.records import Task
from specialist_router.env.reference_agents import OracleAgent, TrivialAgent
from specialist_router.env.tasks import generate_tasks
from specialist_router.env.tools import ToolContext


def _score(agent: Agent, tasks: list[Task], tools: ToolContext, config: Config) -> int:
    correct = 0
    for task in tasks:
        _, verdict = run_and_verify(task, agent, tools, config.episode, config.verifier)
        correct += verdict.correct
    return correct


def test_oracle_100_trivial_0(env_config: Config, tmp_path: Path) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    db_path = tmp_path / "env.sqlite"
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    try:
        oracle_correct = _score(OracleAgent(), tasks, tools, env_config)
        trivial_correct = _score(TrivialAgent(), tasks, tools, env_config)
    finally:
        tools.close()

    assert oracle_correct == len(tasks), f"oracle scored {oracle_correct}/{len(tasks)}"
    assert trivial_correct <= len(tasks) * 0.05, f"trivial scored {trivial_correct}/{len(tasks)}"


class _NeverAnswersAgent:
    """Requests a harmless query every turn but never submits a final answer."""

    name = "never"

    def act(self, obs: Observation) -> Action:
        return ToolRequest("run_sql", {"query": "SELECT 1"})


class _SchemaSpammerAgent:
    """Calls inspect_schema forever to exhaust the tool budget without answering."""

    name = "spammer"

    def act(self, obs: Observation) -> Action:
        return ToolRequest("inspect_schema", {})


def test_episode_stops_at_max_turns(env_config: Config, tmp_path: Path) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    db_path = tmp_path / "env.sqlite"
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    # A high tool budget so max_turns is the binding limit.
    episode_cfg = env_config.episode.model_copy(update={"tool_budget": 1000})
    try:
        traj = run_episode(tasks[0], _NeverAnswersAgent(), tools, episode_cfg)
    finally:
        tools.close()
    assert traj.stop_reason == "max_turns"
    assert traj.num_turns == episode_cfg.max_turns
    assert traj.final_answer is None


def test_episode_stops_at_tool_budget(env_config: Config, tmp_path: Path) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    db_path = tmp_path / "env.sqlite"
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    # Tool budget below max_turns so the budget is the binding limit.
    episode_cfg = env_config.episode.model_copy(update={"tool_budget": 3, "max_turns": 12})
    try:
        traj = run_episode(tasks[0], _SchemaSpammerAgent(), tools, episode_cfg)
    finally:
        tools.close()
    assert traj.stop_reason == "tool_budget"
    assert len(traj.tool_calls) == 3


def test_final_answer_agent_records_trajectory(env_config: Config, tmp_path: Path) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    db_path = tmp_path / "env.sqlite"
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    try:
        traj, verdict = run_and_verify(
            tasks[0], OracleAgent(), tools, env_config.episode, env_config.verifier
        )
    finally:
        tools.close()
    assert traj.stop_reason == "final_answer"
    assert traj.tool_calls  # the oracle ran at least one query
    assert verdict.correct
