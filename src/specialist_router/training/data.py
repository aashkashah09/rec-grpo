"""Env-coupled training data: build the task pool, split it, and sample training tasks.

Training prompts are drawn from the *same* environment as everything else — there is no separate
"training set" of hand-written prompts. :func:`build_task_pool` reuses the Phase-1 generator
(:func:`specialist_router.env.tasks.generate_tasks`); :func:`split_tasks` carves a held-out slice
that the model never trains on; :class:`TaskSampler` draws training tasks, optionally applying a
pass-rate curriculum that skips tasks the model always fails or always solves (both give a
GRPO group near-zero advantage).

The split and the held-out eval sample are deterministic functions of ``task_id`` and a split seed,
so train/held-out membership is stable across runs and never leaks (a task is in exactly one side).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from specialist_router.config import DataSplitConfig, load_config
from specialist_router.env.database import build_dataset
from specialist_router.env.records import Task
from specialist_router.env.tasks import generate_tasks


def build_task_pool(
    env_config_path: str, *, seed_override: int | None = None, n_override: int | None = None
) -> list[Task]:
    """Generate the task pool from an environment config (reusing the Phase-1 generator).

    Args:
        env_config_path: Path to a ``configs/env*.yaml``.
        seed_override: Optional override of the env config seed.
        n_override: Optional override of ``tasks.n_tasks`` (e.g. a small pool for the dry-run).

    Returns:
        The generated tasks, each with programmatic ground truth and reference SQL.
    """
    config = load_config(env_config_path, seed_override=seed_override)
    if n_override is not None:
        config = config.model_copy(
            update={"tasks": config.tasks.model_copy(update={"n_tasks": n_override})}
        )
    dataset = build_dataset(config.db, config.seed)
    return generate_tasks(dataset, config)


@dataclass(frozen=True, slots=True)
class TaskSplit:
    """A disjoint train / held-out partition of the task pool."""

    train: list[Task]
    heldout: list[Task]


def _split_score(task_id: str, split_seed: int) -> float:
    """A deterministic ``[0, 1)`` score for a task, used to assign it to a split reproducibly."""
    digest = hashlib.blake2b(f"{split_seed}:{task_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def split_tasks(tasks: list[Task], heldout_fraction: float, split_seed: int) -> TaskSplit:
    """Partition ``tasks`` into train/held-out, balanced per template and fully deterministic.

    Within each template the tasks are ranked by a seeded hash of their ``task_id`` and the lowest
    ``ceil(heldout_fraction * n)`` go to held-out. Balancing per template keeps every template
    represented on both sides (so per-template eval is meaningful) regardless of the pool's order.

    Args:
        tasks: The full task pool.
        heldout_fraction: Fraction of each template's tasks to hold out (``0 < f < 1``).
        split_seed: Seed making the hash-based assignment reproducible.

    Returns:
        A :class:`TaskSplit`; ``train`` and ``heldout`` preserve the input order.

    Raises:
        ValueError: If ``heldout_fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be in (0, 1), got {heldout_fraction}")

    by_template: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_template[task.template_id].append(task)

    heldout_ids: set[str] = set()
    for template_tasks in by_template.values():
        ranked = sorted(template_tasks, key=lambda t: _split_score(t.task_id, split_seed))
        n_heldout = int(np.ceil(heldout_fraction * len(ranked)))
        heldout_ids.update(t.task_id for t in ranked[:n_heldout])

    train = [t for t in tasks if t.task_id not in heldout_ids]
    heldout = [t for t in tasks if t.task_id in heldout_ids]
    return TaskSplit(train=train, heldout=heldout)


def heldout_eval_sample(heldout: list[Task], n: int, split_seed: int) -> list[Task]:
    """A fixed, template-balanced sample of ``n`` held-out tasks for stable eval curves.

    Deterministic (seeded hash), so every eval step scores the *same* tasks and the success curve
    reflects the model, not sampling noise. Draws round-robin across templates so the sample stays
    balanced even when ``n`` is small.

    Args:
        heldout: The held-out tasks.
        n: Target sample size (clamped to ``len(heldout)``).
        split_seed: Seed for the deterministic ordering.

    Returns:
        Up to ``n`` held-out tasks.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    by_template: dict[str, list[Task]] = defaultdict(list)
    for task in sorted(heldout, key=lambda t: _split_score(t.task_id, split_seed + 1)):
        by_template[task.template_id].append(task)

    queues = [by_template[k] for k in sorted(by_template)]
    sample: list[Task] = []
    idx = 0
    while len(sample) < min(n, len(heldout)):
        queue = queues[idx % len(queues)]
        if queue:
            sample.append(queue.pop(0))
        idx += 1
        if idx > len(heldout) * 2:  # safety: all queues drained
            break
    return sample


@dataclass(slots=True)
class _PassStats:
    """Running success stats for one task (for the pass-rate curriculum)."""

    n: int = 0
    n_correct: int = 0

    @property
    def pass_rate(self) -> float:
        """Empirical pass rate, or ``0.5`` (neutral) before any observation."""
        return self.n_correct / self.n if self.n else 0.5


@dataclass(slots=True)
class TaskSampler:
    """Draws training tasks, optionally applying a pass-rate curriculum band.

    ``record_outcome`` feeds verifier verdicts back so that, once a task has at least
    ``min_observations`` rollouts, it is skipped while its pass rate sits outside
    ``[min_pass_rate, max_pass_rate]`` (always-failed or always-solved). If the band would empty the
    eligible pool, sampling falls back to the full training set — the curriculum narrows, never
    starves, the sampler.
    """

    tasks: list[Task]
    rng: np.random.Generator
    min_pass_rate: float
    max_pass_rate: float
    min_observations: int
    _stats: dict[str, _PassStats] = field(default_factory=dict)

    @classmethod
    def from_config(cls, tasks: list[Task], config: DataSplitConfig, seed: int) -> TaskSampler:
        """Build a sampler over ``tasks`` from the data-split config."""
        return cls(
            tasks=list(tasks),
            rng=np.random.default_rng(seed),
            min_pass_rate=config.curriculum_min_pass_rate,
            max_pass_rate=config.curriculum_max_pass_rate,
            min_observations=config.curriculum_min_observations,
        )

    def record_outcome(self, task_id: str, correct: bool) -> None:
        """Record one rollout's verdict for the curriculum's pass-rate estimate."""
        stats = self._stats.setdefault(task_id, _PassStats())
        stats.n += 1
        stats.n_correct += int(bool(correct))

    def _in_band(self, task: Task) -> bool:
        """Whether a task is currently eligible under the curriculum band."""
        stats = self._stats.get(task.task_id)
        if stats is None or stats.n < self.min_observations:
            return True  # not enough evidence yet — keep it in play
        return self.min_pass_rate <= stats.pass_rate <= self.max_pass_rate

    def sample(self, n: int) -> list[Task]:
        """Sample ``n`` training tasks (with replacement) from the eligible pool.

        Args:
            n: Number of tasks to draw (one per GRPO group in the next rollout batch).

        Returns:
            ``n`` tasks; drawn from the in-band pool, or the full pool if the band is empty.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        eligible = [t for t in self.tasks if self._in_band(t)] or self.tasks
        indices = self.rng.integers(0, len(eligible), size=n)
        return [eligible[int(i)] for i in indices]
