"""CPU dry-run: exercise the full GRPO pipeline with mocked generation and a mock trainer step.

This is the CI-default validation of the training wiring (ADR-012). It uses **no** torch/trl/vLLM:
generation is a scripted mock, but everything downstream is real — the sandboxed episode loop, the
Phase-1 verifier, the training reward, the per-episode reward cache, and the GRPO group-relative
advantage. A :class:`MockTrainer` asserts the exact input contract a real ``GRPOTrainer`` step
consumes (aligned ``completion_ids``/``mask``/``logprobs``, non-empty prompts, an advantage per
episode), so the wiring can be validated locally and in CI before any GPU is rented.

The opt-in *real* tiny-model step (``@pytest.mark.training`` / ``make grpo-dryrun``) lives behind
the ``training`` extra; this module deliberately does not import it.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from specialist_router.config import GrpoConfig, load_config
from specialist_router.env.database import build_dataset, write_sqlite_file
from specialist_router.env.records import AnswerType, AnswerValue, Task
from specialist_router.env.tools import ToolContext
from specialist_router.training.data import TaskSampler, build_task_pool, split_tasks
from specialist_router.training.rollout import EnvRollout, EpisodeRollout, TurnGeneration

_ADVANTAGE_EPS = 1e-8


def mock_encode(text: str) -> list[int]:
    """A deterministic, dependency-free tokenizer stand-in (per-whitespace-token byte sums).

    Only the token *count* and determinism matter for validating masks/logprob alignment; the real
    run injects the model tokenizer. Always returns at least one id so empty strings still occupy a
    (masked) position.
    """
    words = text.split() or [text]
    return [sum(word.encode("utf-8")) % 997 + 1 for word in words]


def _wrong_answer(answer_type: AnswerType) -> AnswerValue:
    """A shape-valid but incorrect answer, so an episode scores 0 through the real verifier."""
    if answer_type is AnswerType.LIST_STR:
        return ["__wrong__"]
    if answer_type is AnswerType.INTEGER:
        return -1
    return -999999.0


class MockPolicy:
    """A scripted, task-aware generation mock producing mixed-correctness episodes.

    Per episode it emits ``run_sql(reference_sql)`` then a ``final_answer``. To create the
    intra-group variance GRPO needs (a group whose G episodes are all identical has ~0 advantage),
    successive episodes of the *same* task alternate correct / wrong answers — so every group mixes
    rewards. The mock identifies the current task from the ``Task:`` line the system prompt embeds.
    """

    def __init__(self, tasks: list[Task]) -> None:
        """Index the batch tasks by their question text (unique per task)."""
        self._task_by_question = {t.question: t for t in tasks}
        self._episode_count: Counter[str] = Counter()

    def __call__(self, messages: list[dict[str, str]], max_new_tokens: int) -> TurnGeneration:
        """Emit the next scripted turn as a JSON action."""
        question = messages[0]["content"].split("Task: ", 1)[-1]
        task = self._task_by_question[question]
        first_turn = not any(m["role"] == "assistant" for m in messages)
        if first_turn:
            self._episode_count[question] += 1
            payload: dict[str, object] = {
                "tool": "run_sql",
                "arguments": {"query": task.reference_sql},
            }
        else:
            episode_index = self._episode_count[question] - 1
            correct = episode_index % 2 == 0
            value = task.expected if correct else _wrong_answer(task.answer_type)
            payload = {"final_answer": value}
        text = json.dumps(payload)
        return TurnGeneration(text=text, token_ids=mock_encode(text), logprobs=_mock_logprobs(text))


def _mock_logprobs(text: str) -> list[float]:
    """One negative logprob per mock token (values are placeholders; only alignment is checked)."""
    return [-0.5 for _ in mock_encode(text)]


def group_advantages(rollouts: list[EpisodeRollout]) -> list[float]:
    """Compute GRPO group-relative advantages ``(r - mean)/(std + eps)`` per task group."""
    by_task: dict[str, list[int]] = {}
    for i, roll in enumerate(rollouts):
        by_task.setdefault(roll.task_id, []).append(i)
    advantages = [0.0] * len(rollouts)
    for indices in by_task.values():
        rewards = np.array([rollouts[i].reward.reward for i in indices], dtype=float)
        centered = rewards - rewards.mean()
        scaled = centered / (rewards.std() + _ADVANTAGE_EPS)
        for pos, i in enumerate(indices):
            advantages[i] = float(scaled[pos])
    return advantages


@dataclass(frozen=True, slots=True)
class MockStepResult:
    """What one mock trainer step observed (used by the summary and the CI assertions)."""

    n_episodes: int
    n_groups: int
    n_groups_with_signal: int
    mean_reward: float
    mean_correct: float
    mean_format: float
    total_assistant_tokens: int


class MockTrainer:
    """Asserts the GRPOTrainer input contract on a batch of rollouts and computes advantages.

    A real step optimizes the policy; this validates that the rollout produced exactly what such a
    step consumes, so a wiring bug surfaces on CPU rather than after renting a GPU.
    """

    def step(self, rollouts: list[EpisodeRollout]) -> MockStepResult:
        """Validate one batch and return its aggregate statistics."""
        if not rollouts:
            raise ValueError("mock trainer step received no rollouts")
        total_assistant = 0
        for roll in rollouts:
            n = len(roll.completion_ids)
            if not (len(roll.completion_mask) == n == len(roll.logprobs)):
                raise ValueError(
                    f"episode {roll.episode_id}: completion_ids/mask/logprobs lengths differ"
                )
            if not roll.prompt_ids:
                raise ValueError(f"episode {roll.episode_id}: empty prompt_ids")
            if any(bit not in (0, 1) for bit in roll.completion_mask):
                raise ValueError(f"episode {roll.episode_id}: completion_mask is not binary")
            total_assistant += sum(roll.completion_mask)

        advantages = group_advantages(rollouts)
        if len(advantages) != len(rollouts):
            raise ValueError("advantage count does not match episode count")

        by_task: Counter[str] = Counter(r.task_id for r in rollouts)
        signal = sum(1 for a in _grouped(advantages, rollouts) if a)
        return MockStepResult(
            n_episodes=len(rollouts),
            n_groups=len(by_task),
            n_groups_with_signal=signal,
            mean_reward=float(np.mean([r.reward.reward for r in rollouts])),
            mean_correct=float(np.mean([r.reward.correct for r in rollouts])),
            mean_format=float(np.mean([r.reward.format_score for r in rollouts])),
            total_assistant_tokens=total_assistant,
        )


def _grouped(advantages: list[float], rollouts: list[EpisodeRollout]) -> list[bool]:
    """Whether each task group has any nonzero advantage (i.e. GRPO gets a gradient from it)."""
    by_task: dict[str, list[float]] = {}
    for adv, roll in zip(advantages, rollouts, strict=True):
        by_task.setdefault(roll.task_id, []).append(adv)
    return [any(abs(a) > _ADVANTAGE_EPS for a in advs) for advs in by_task.values()]


@dataclass(frozen=True, slots=True)
class DryRunSummary:
    """Aggregate result of the whole mocked dry-run (returned so the CLI/tests can assert on it)."""

    steps: int
    n_episodes: int
    n_groups_with_signal: int
    mean_reward: float
    mean_correct: float
    mean_format: float

    def render(self) -> str:
        """A short human-readable summary."""
        return (
            f"dry-run OK: steps={self.steps} episodes={self.n_episodes} "
            f"groups_with_gradient={self.n_groups_with_signal} "
            f"mean_reward={self.mean_reward:.3f} mean_correct={self.mean_correct:.3f} "
            f"mean_format={self.mean_format:.3f}"
        )


def run_mock_dry_run(config: GrpoConfig) -> DryRunSummary:
    """Run the full mocked GRPO pipeline on CPU and return a summary.

    Exercises: env task generation -> curriculum sampler -> env-coupled rollout (real episode loop +
    sandbox) -> real verifier -> training reward -> per-episode reward cache -> reward_fn lookup ->
    group-relative advantages -> mock trainer-step contract check. No GPU, no heavy deps.

    Args:
        config: The GRPO config (its ``dry_run`` block controls pool size, group size, and steps).

    Returns:
        A :class:`DryRunSummary`.
    """
    env = load_config(config.dry_run.env_config)
    dataset = build_dataset(env.db, env.seed)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "dryrun.sqlite")
        write_sqlite_file(dataset, db_path)
        tools = ToolContext(db_path, env.tools.run_sql, env.tools.python_calc)
        try:
            pool = build_task_pool(config.dry_run.env_config, n_override=config.dry_run.n_tasks * 4)
            split = split_tasks(pool, config.data.heldout_fraction, config.data.split_seed)
            sampler = TaskSampler.from_config(split.train, config.data, config.seed)
            trainer = MockTrainer()

            steps: list[MockStepResult] = []
            for _ in range(config.dry_run.max_steps):
                unique = sampler.sample(config.dry_run.n_tasks)
                batch = [t for t in unique for _ in range(config.dry_run.num_generations)]
                policy = MockPolicy(batch)
                rollout = EnvRollout(
                    generate_fn=policy,
                    encode=mock_encode,
                    tools=tools,
                    rollout_config=config.rollout,
                    verifier_config=env.verifier,
                    reward_config=config.reward,
                )
                episodes = rollout.generate_batch(batch)
                _assert_reward_fn_roundtrip(rollout, episodes, batch)
                for roll in episodes:
                    sampler.record_outcome(roll.task_id, bool(roll.verdict.correct))
                steps.append(trainer.step(episodes))
        finally:
            tools.close()

    total_episodes = sum(s.n_episodes for s in steps)
    return DryRunSummary(
        steps=len(steps),
        n_episodes=total_episodes,
        n_groups_with_signal=sum(s.n_groups_with_signal for s in steps),
        mean_reward=float(np.mean([s.mean_reward for s in steps])),
        mean_correct=float(np.mean([s.mean_correct for s in steps])),
        mean_format=float(np.mean([s.mean_format for s in steps])),
    )


def _assert_reward_fn_roundtrip(
    rollout: EnvRollout, episodes: list[EpisodeRollout], batch: list[Task]
) -> None:
    """Check that reward_fn reproduces each episode's cached reward in batch order (rider 1).

    Two episodes in a group can emit identical completions; because the reward cache is keyed by the
    minted per-episode id (not completion content), reward_fn must still return each episode's own
    reward. This drives that path exactly as TRL would.
    """
    rewards = rollout.reward_fn(
        completions=[e.completion_ids for e in episodes],
        task_id=[t.task_id for t in batch],
    )
    expected = [e.reward.reward for e in episodes]
    if rewards != expected:
        raise ValueError("reward_fn did not reproduce cached per-episode rewards in batch order")
