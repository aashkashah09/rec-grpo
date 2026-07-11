"""Held-out evaluation: task success and per-template breakdown over the Phase-1 verifier.

This is the single evaluation path shared by the GRPO held-out eval callback (every K steps) and by
Phase-4 integration. It never introduces a new verification route: it drives an agent through the
same sandboxed episode loop and scores it with :func:`specialist_router.env.episode.run_and_verify`
(PROJECT_PLAN §1.2 — the specialist is judged by held-out task success, never by trajectory-level
importance sampling).

Alongside success it reports a **format-compliance rate** (how well-formed the episodes are,
independent of correctness). Tracking it next to success is a drift guard: format-rate falling while
reward/success rises is a warning of protocol drift or verifier gaming (see ADR-011 and the Phase-3
run notes).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from specialist_router.agents.base import Agent
from specialist_router.config import EpisodeConfig, VerifierConfig
from specialist_router.env.episode import run_and_verify
from specialist_router.env.records import Task
from specialist_router.env.tools import ToolContext
from specialist_router.training.reward import (
    FORMAT_COMPONENTS,
    format_score,
    signals_from_trajectory,
)

AgentFactory = Callable[[], Agent]
"""A zero-arg factory returning a fresh agent per episode (agents hold per-episode state)."""


@dataclass(frozen=True, slots=True)
class TemplateStat:
    """Success and format-compliance for one task template."""

    n: int
    success: float
    format_rate: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The result of one held-out evaluation pass."""

    n_tasks: int
    overall_success: float
    format_rate: float
    per_template: dict[str, TemplateStat]
    per_difficulty: dict[str, float]

    def summary(self) -> str:
        """A compact one-line-per-template summary for logs."""
        lines = [
            f"overall_success={self.overall_success:.3f} "
            f"format_rate={self.format_rate:.3f} (n={self.n_tasks})"
        ]
        for template_id in sorted(self.per_template):
            stat = self.per_template[template_id]
            lines.append(
                f"  {template_id}: success={stat.success:.3f} "
                f"format_rate={stat.format_rate:.3f} (n={stat.n})"
            )
        return "\n".join(lines)


def evaluate(
    agent_factory: AgentFactory,
    tasks: list[Task],
    tools: ToolContext,
    episode_config: EpisodeConfig,
    verifier_config: VerifierConfig,
) -> EvalReport:
    """Run ``agent_factory`` over ``tasks`` and report success + format compliance.

    Args:
        agent_factory: Builds a fresh agent per episode.
        tasks: The held-out tasks to evaluate.
        tools: The sandboxed tool context bound to the task database.
        episode_config: Turn / tool budgets.
        verifier_config: Numeric tolerances for the verifier.

    Returns:
        An :class:`EvalReport` with overall and per-template/per-difficulty breakdowns.

    Raises:
        ValueError: If ``tasks`` is empty.
    """
    if not tasks:
        raise ValueError("evaluate requires at least one task")

    template_n: dict[str, int] = defaultdict(int)
    template_correct: dict[str, float] = defaultdict(float)
    template_format: dict[str, float] = defaultdict(float)
    difficulty_n: dict[str, int] = defaultdict(int)
    difficulty_correct: dict[str, float] = defaultdict(float)
    total_correct = 0.0
    total_format = 0.0

    components = list(FORMAT_COMPONENTS)
    for task in tasks:
        agent = agent_factory()
        trajectory, verdict = run_and_verify(task, agent, tools, episode_config, verifier_config)
        correct = 1.0 if verdict.correct else 0.0
        fmt = format_score(signals_from_trajectory(trajectory), components)

        total_correct += correct
        total_format += fmt
        template_n[task.template_id] += 1
        template_correct[task.template_id] += correct
        template_format[task.template_id] += fmt
        difficulty_n[task.difficulty] += 1
        difficulty_correct[task.difficulty] += correct

    n = len(tasks)
    per_template = {
        tid: TemplateStat(
            n=template_n[tid],
            success=template_correct[tid] / template_n[tid],
            format_rate=template_format[tid] / template_n[tid],
        )
        for tid in template_n
    }
    per_difficulty = {diff: difficulty_correct[diff] / difficulty_n[diff] for diff in difficulty_n}
    return EvalReport(
        n_tasks=n,
        overall_success=total_correct / n,
        format_rate=total_format / n,
        per_template=per_template,
        per_difficulty=per_difficulty,
    )
