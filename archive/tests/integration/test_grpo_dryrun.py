"""Integration tests for the GRPO rollout, reward cache, and the CPU dry-run pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from specialist_router.config import Config, load_grpo_config
from specialist_router.env.database import Dataset, write_sqlite_file
from specialist_router.env.records import Task
from specialist_router.env.tasks import generate_tasks
from specialist_router.env.tools import ToolContext
from specialist_router.training.dry_run import (
    MockPolicy,
    MockTrainer,
    group_advantages,
    mock_encode,
    run_mock_dry_run,
)
from specialist_router.training.rollout import EnvRollout, TurnGeneration

_GRPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "grpo.yaml"


def _tasks(mini_dataset: Dataset, env_config: Config, n: int) -> list[Task]:
    cfg = env_config.model_copy(
        update={"tasks": env_config.tasks.model_copy(update={"n_tasks": n})}
    )
    return generate_tasks(mini_dataset, cfg)


def _rollout(mini_dataset: Dataset, env_config: Config, tmp_path: Path, tasks: list[Task]):
    db_path = str(tmp_path / "env.sqlite")
    write_sqlite_file(mini_dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    grpo = load_grpo_config(_GRPO_CONFIG)
    policy = MockPolicy(tasks)
    return (
        EnvRollout(
            generate_fn=policy,
            encode=mock_encode,
            tools=tools,
            rollout_config=grpo.rollout,
            verifier_config=env_config.verifier,
            reward_config=grpo.reward,
        ),
        tools,
    )


def test_rollout_masks_align_and_mark_assistant_tokens(
    mini_dataset: Dataset, env_config: Config, tmp_path: Path
) -> None:
    tasks = _tasks(mini_dataset, env_config, 4)[:2]
    rollout, tools = _rollout(mini_dataset, env_config, tmp_path, tasks)
    try:
        episodes = rollout.generate_batch(tasks)
    finally:
        tools.close()

    for ep in episodes:
        n = len(ep.completion_ids)
        assert len(ep.completion_mask) == n == len(ep.logprobs)  # three lists share one length
        assert set(ep.completion_mask) <= {0, 1}
        assert ep.prompt_ids, "prompt must be non-empty"
        assert sum(ep.completion_mask) > 0, "at least one assistant (trained) token"
        # The scripted episode calls run_sql then answers, so an injected tool-result span exists.
        assert 0 in ep.completion_mask, "injected tool-result tokens must be masked out"
        # Masked-out logprobs are zeroed; masked-in ones carry the (mock) generation logprob.
        for bit, lp in zip(ep.completion_mask, ep.logprobs, strict=True):
            if bit == 0:
                assert lp == 0.0


def test_reward_cache_keyed_by_episode_id_not_completion(
    mini_dataset: Dataset, env_config: Config, tmp_path: Path
) -> None:
    """Rider 1: identical completions within a group must not collide in the reward cache."""
    task = _tasks(mini_dataset, env_config, 1)[0]

    class _AlwaysCorrect:
        """Deterministic policy: every episode of a task emits the identical correct transcript."""

        def __call__(self, messages: list[dict[str, str]], max_new_tokens: int) -> TurnGeneration:
            import json

            if not any(m["role"] == "assistant" for m in messages):
                text = json.dumps({"tool": "run_sql", "arguments": {"query": task.reference_sql}})
            else:
                text = json.dumps({"final_answer": task.expected})
            return TurnGeneration(text, mock_encode(text), [-0.5] * len(mock_encode(text)))

    db_path = str(tmp_path / "env.sqlite")
    write_sqlite_file(mini_dataset, db_path)
    tools = ToolContext(db_path, env_config.tools.run_sql, env_config.tools.python_calc)
    grpo = load_grpo_config(_GRPO_CONFIG)
    rollout = EnvRollout(
        generate_fn=_AlwaysCorrect(),
        encode=mock_encode,
        tools=tools,
        rollout_config=grpo.rollout,
        verifier_config=env_config.verifier,
        reward_config=grpo.reward,
    )
    batch = [task, task]  # two episodes of ONE task -> a group of 2
    try:
        episodes = rollout.generate_batch(batch)
        rewards = rollout.reward_fn(
            completions=[e.completion_ids for e in episodes],
            task_id=[task.task_id, task.task_id],
        )
    finally:
        tools.close()

    # The two completions are byte-identical...
    assert episodes[0].completion_ids == episodes[1].completion_ids
    # ...but the episodes carry distinct minted ids, so the cache holds two entries...
    assert episodes[0].episode_id != episodes[1].episode_id
    assert episodes[0].episode_id == f"{task.task_id}#0"
    assert episodes[1].episode_id == f"{task.task_id}#1"
    # ...and reward_fn returns one reward per position (not one collapsed entry).
    assert len(rewards) == 2
    assert rewards == [e.reward.reward for e in episodes]


def test_reward_fn_requires_task_id(
    mini_dataset: Dataset, env_config: Config, tmp_path: Path
) -> None:
    tasks = _tasks(mini_dataset, env_config, 2)
    rollout, tools = _rollout(mini_dataset, env_config, tmp_path, tasks)
    try:
        rollout.generate_batch(tasks)
        with pytest.raises(ValueError, match="task_id"):
            rollout.reward_fn(completions=[[1], [2]])
    finally:
        tools.close()


def test_group_advantages_zero_when_group_uniform() -> None:
    from specialist_router.env.records import Trajectory, Verdict
    from specialist_router.training.reward import RewardParts
    from specialist_router.training.rollout import EpisodeRollout

    def _ep(task_id: str, reward: float) -> EpisodeRollout:
        return EpisodeRollout(
            episode_id=f"{task_id}#x",
            task_id=task_id,
            prompt_ids=[1],
            completion_ids=[1],
            completion_mask=[1],
            logprobs=[0.0],
            trajectory=Trajectory(task_id=task_id, stop_reason="final_answer", num_turns=1),
            verdict=Verdict(
                correct=True,
                reason="",
                answer_type="integer",  # type: ignore[arg-type]
                extracted=1,
                expected=1,
                tolerance={},
            ),
            reward=RewardParts(correct=1, format_score=1.0, reward=reward),
        )

    # A uniform group (all same reward) -> ~0 advantage (binary collapse); a mixed group -> nonzero.
    uniform = [_ep("a", 1.0), _ep("a", 1.0)]
    mixed = [_ep("b", 1.0), _ep("b", 0.0)]
    assert all(abs(a) < 1e-6 for a in group_advantages(uniform))
    assert any(abs(a) > 1e-6 for a in group_advantages(mixed))


def test_mock_trainer_rejects_misaligned_batch() -> None:
    from specialist_router.env.records import Trajectory, Verdict
    from specialist_router.training.reward import RewardParts
    from specialist_router.training.rollout import EpisodeRollout

    bad = EpisodeRollout(
        episode_id="x#0",
        task_id="x",
        prompt_ids=[1],
        completion_ids=[1, 2],
        completion_mask=[1],  # length mismatch
        logprobs=[0.0, 0.0],
        trajectory=Trajectory(task_id="x", stop_reason="final_answer", num_turns=1),
        verdict=Verdict(
            correct=False,
            reason="",
            answer_type="integer",  # type: ignore[arg-type]
            extracted=None,
            expected=1,
            tolerance={},
        ),
        reward=RewardParts(correct=0, format_score=0.0, reward=0.0),
    )
    with pytest.raises(ValueError, match="lengths differ"):
        MockTrainer().step([bad])


def test_dry_run_end_to_end_produces_gradient_signal() -> None:
    config = load_grpo_config(_GRPO_CONFIG, seed_override=0)
    summary = run_mock_dry_run(config)
    assert summary.steps == config.dry_run.max_steps
    expected_episodes = (
        config.dry_run.max_steps * config.dry_run.n_tasks * config.dry_run.num_generations
    )
    assert summary.n_episodes == expected_episodes
    # Mixed-correctness groups give GRPO a nonzero advantage on at least some groups.
    assert summary.n_groups_with_signal > 0
    assert 0.0 <= summary.mean_correct <= 1.0
