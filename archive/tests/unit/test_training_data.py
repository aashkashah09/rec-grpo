"""Unit tests for the training data split, held-out sampling, and the curriculum sampler."""

from __future__ import annotations

import numpy as np

from specialist_router.config import Config
from specialist_router.env.database import Dataset
from specialist_router.env.tasks import generate_tasks
from specialist_router.training.data import (
    TaskSampler,
    heldout_eval_sample,
    split_tasks,
)


def _tasks(mini_dataset: Dataset, env_config: Config, n: int) -> list:
    cfg = env_config.model_copy(
        update={"tasks": env_config.tasks.model_copy(update={"n_tasks": n})}
    )
    return generate_tasks(mini_dataset, cfg)


def test_split_is_deterministic_and_disjoint(mini_dataset: Dataset, env_config: Config) -> None:
    tasks = _tasks(mini_dataset, env_config, 80)
    a = split_tasks(tasks, 0.25, split_seed=7)
    b = split_tasks(tasks, 0.25, split_seed=7)
    assert [t.task_id for t in a.heldout] == [t.task_id for t in b.heldout]  # deterministic

    train_ids = {t.task_id for t in a.train}
    heldout_ids = {t.task_id for t in a.heldout}
    assert train_ids.isdisjoint(heldout_ids)  # a task is on exactly one side
    assert train_ids | heldout_ids == {t.task_id for t in tasks}


def test_split_is_template_balanced(mini_dataset: Dataset, env_config: Config) -> None:
    tasks = _tasks(mini_dataset, env_config, 80)
    split = split_tasks(tasks, 0.25, split_seed=7)
    templates = {t.template_id for t in tasks}
    heldout_templates = {t.template_id for t in split.heldout}
    # Every template is represented in the held-out set (per-template eval is meaningful).
    assert heldout_templates == templates


def test_split_seed_changes_membership(mini_dataset: Dataset, env_config: Config) -> None:
    tasks = _tasks(mini_dataset, env_config, 80)
    a = split_tasks(tasks, 0.25, split_seed=1)
    b = split_tasks(tasks, 0.25, split_seed=2)
    assert {t.task_id for t in a.heldout} != {t.task_id for t in b.heldout}


def test_heldout_eval_sample_is_fixed_and_balanced(
    mini_dataset: Dataset, env_config: Config
) -> None:
    tasks = _tasks(mini_dataset, env_config, 80)
    heldout = split_tasks(tasks, 0.4, split_seed=3).heldout
    s1 = heldout_eval_sample(heldout, 16, split_seed=3)
    s2 = heldout_eval_sample(heldout, 16, split_seed=3)
    assert [t.task_id for t in s1] == [t.task_id for t in s2]  # fixed across calls
    assert len(s1) == 16
    # Round-robin across templates keeps the small sample balanced.
    counts = {}
    for task in s1:
        counts[task.template_id] = counts.get(task.template_id, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sampler_curriculum_skips_always_solved(mini_dataset: Dataset, env_config: Config) -> None:
    tasks = _tasks(mini_dataset, env_config, 40)
    sampler = TaskSampler(
        tasks=tasks,
        rng=np.random.default_rng(0),
        min_pass_rate=0.05,
        max_pass_rate=0.95,
        min_observations=4,
    )
    solved = tasks[0]
    for _ in range(8):  # record it as always solved (pass rate 1.0 > max 0.95)
        sampler.record_outcome(solved.task_id, correct=True)
    drawn = {t.task_id for t in sampler.sample(500)}
    assert solved.task_id not in drawn  # curriculum drops the always-solved task
    assert len(drawn) > 1  # but keeps the rest in play


def test_sampler_falls_back_when_band_empty(mini_dataset: Dataset, env_config: Config) -> None:
    tasks = _tasks(mini_dataset, env_config, 8)
    sampler = TaskSampler(
        tasks=tasks,
        rng=np.random.default_rng(0),
        min_pass_rate=0.05,
        max_pass_rate=0.95,
        min_observations=2,
    )
    for task in tasks:  # every task always solved -> band would be empty
        for _ in range(4):
            sampler.record_outcome(task.task_id, correct=True)
    drawn = sampler.sample(10)
    assert len(drawn) == 10  # never starves: falls back to the full pool
