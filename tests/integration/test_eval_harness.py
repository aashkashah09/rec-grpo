"""Integration tests for the shared held-out evaluation harness."""

from __future__ import annotations

from pathlib import Path

from specialist_router.config import Config
from specialist_router.env.database import Dataset, write_sqlite_file
from specialist_router.env.reference_agents import OracleAgent, TrivialAgent
from specialist_router.env.tasks import generate_tasks
from specialist_router.env.tools import ToolContext
from specialist_router.evaluation.harness import evaluate


def _tasks(mini_dataset: Dataset, env_config: Config, n: int) -> list:
    cfg = env_config.model_copy(
        update={"tasks": env_config.tasks.model_copy(update={"n_tasks": n})}
    )
    return generate_tasks(mini_dataset, cfg)


def test_oracle_scores_perfect_with_per_template_breakdown(
    mini_dataset: Dataset, env_config: Config, tmp_path: Path
) -> None:
    tasks = _tasks(mini_dataset, env_config, 24)
    db_path = str(tmp_path / "env.sqlite")
    write_sqlite_file(mini_dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    try:
        report = evaluate(OracleAgent, tasks, tools, env_config.episode, env_config.verifier)
    finally:
        tools.close()

    assert report.overall_success == 1.0
    assert report.n_tasks == len(tasks)
    assert report.per_template  # broken down by template
    assert all(stat.success == 1.0 for stat in report.per_template.values())
    assert "\n" in report.summary()


def test_format_rate_is_independent_of_correctness(
    mini_dataset: Dataset, env_config: Config, tmp_path: Path
) -> None:
    """The trivial agent answers well-formedly but wrongly: success ~0, format-rate still high.

    This is exactly what makes the rider-2 drift guard meaningful — format-rate measures protocol
    compliance, not correctness, so the two curves can move independently.
    """
    tasks = _tasks(mini_dataset, env_config, 16)
    db_path = str(tmp_path / "env.sqlite")
    write_sqlite_file(mini_dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    try:
        report = evaluate(TrivialAgent, tasks, tools, env_config.episode, env_config.verifier)
    finally:
        tools.close()

    assert report.overall_success == 0.0  # trivial agent is always wrong
    assert report.format_rate > 0.0  # yet it submits well-formed final answers
