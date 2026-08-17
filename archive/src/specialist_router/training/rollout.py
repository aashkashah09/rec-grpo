"""Env-coupled multi-turn GRPO rollout: generate tool-use episodes and score them by the verifier.

This is the body behind TRL's ``rollout_func`` (ADR-010). For each training task we run a full
multi-turn episode against the *real* sandboxed environment, capture the exact token stream the
policy emitted (with a mask that marks assistant tokens vs. injected tool-result tokens), verify
the final answer with the Phase-1 verifier, and turn the verdict into the per-episode GRPO reward.

Design choices that keep this testable on CPU without any heavy dependency:

* **All control flow is reused, not reimplemented.** The 12-turn / tool-budget invariants, the tool
  dispatch, and the :class:`~specialist_router.env.records.Trajectory` all come from
  :func:`specialist_router.env.episode.run_episode`. A :class:`_RecordingAgent` plugs into that loop
  and records the token stream as a side effect, so budget/dispatch/verify logic has exactly one
  home.
* **Generation and tokenization are injected** (``generate_fn`` / ``encode``). Production binds them
  to vLLM + the model tokenizer; the CPU dry-run and tests bind them to mocks. This module never
  imports torch/transformers/vllm.
* **The reward cache is keyed by a unique per-episode id minted here** — ``f"{task_id}#{k}"`` for
  the ``k``-th episode of that task in the batch — *not* by completion token ids. Two episodes in
  one GRPO group can emit identical completions; a content-derived key would collide and cross-wire
  their rewards. The reward function reconstructs the same id from the batch's ``task_id`` column in
  order, so no custom field has to survive TRL's rollout→reward boundary.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from specialist_router.agents.base import Action, Observation
from specialist_router.agents.chat_agent import _parse_action
from specialist_router.config import (
    EpisodeConfig,
    RolloutConfig,
    TrainingRewardConfig,
    VerifierConfig,
)
from specialist_router.env.episode import run_episode, system_prompt
from specialist_router.env.records import AnswerType, Task, Trajectory, Verdict
from specialist_router.env.tools import ToolContext
from specialist_router.env.verifier import verify
from specialist_router.training.reward import RewardParts, episode_reward, signals_from_trajectory

_ACTION_INSTRUCTIONS = (
    "Respond with exactly one JSON object and nothing else. To call a tool: "
    '{"tool": "run_sql", "arguments": {"query": "SELECT ..."}} '
    "(tools: inspect_schema, run_sql, python_calc). To answer: "
    '{"final_answer": <value matching the task\'s answer format>}.'
)
"""Kept identical to :data:`specialist_router.agents.chat_agent._ACTION_INSTRUCTIONS` so the RL
rollout, the Phase-2 live arms, and the eval harness all speak one tool protocol (ADR-010)."""


@dataclass(frozen=True, slots=True)
class TurnGeneration:
    """One assistant turn's decoded text plus the token ids and per-token logprobs that produced it.

    In production this comes from vLLM; in the dry-run/tests from a mock. ``logprobs`` must be
    aligned one-to-one with ``token_ids`` (the values GRPO differentiates through).
    """

    text: str
    token_ids: list[int]
    logprobs: list[float]


# Inject-able generation and tokenization (bound to vLLM + tokenizer in production, mocks in tests).
GenerateFn = Callable[[list[dict[str, str]], int], TurnGeneration]
Encode = Callable[[str], list[int]]


@dataclass(frozen=True, slots=True)
class EpisodeRollout:
    """One episode's tokenized rollout and its verifier-derived reward.

    The token layout is ``prompt_ids`` (the initial system+instructions context) followed by
    ``completion_ids`` = ``[assistant turn 1][tool result 1][assistant turn 2]…``. The
    ``completion_mask`` is ``1`` on assistant (policy) tokens and ``0`` on injected tokens, so the
    loss trains only what the model generated. ``logprobs`` aligns with ``completion_ids`` (``0.0``
    on masked positions). The three completion lists always share one length.
    """

    episode_id: str
    task_id: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    logprobs: list[float]
    trajectory: Trajectory
    verdict: Verdict
    reward: RewardParts


def _generation_parsed(text: str) -> bool:
    """Whether a raw generation parses to a valid action under the chat-agent tool protocol.

    Mirrors the acceptance condition of
    :func:`specialist_router.agents.chat_agent._parse_action` (first ``{…}`` is a JSON object that
    is either a ``final_answer`` or a ``{"tool": str}`` call). Used to measure the precise
    ``all_actions_parse`` format signal, which a recorded :class:`Trajectory` alone cannot recover.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and ("final_answer" in obj or isinstance(obj.get("tool"), str))


@dataclass(slots=True)
class _RecordingAgent:
    """An episode agent that generates via ``generate_fn`` and records the token stream + masks.

    It plugs into :func:`run_episode` (which owns the budgets, dispatch, and trajectory) and mirrors
    :class:`~specialist_router.agents.chat_agent.ChatToolAgent`'s message construction so training
    rollouts and the Phase-2 live arms are byte-compatible in prompt shape.
    """

    generate_fn: GenerateFn
    encode: Encode
    answer_type: AnswerType
    max_new_tokens: int
    max_tool_output_chars: int
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    completion_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    parsed_flags: list[bool] = field(default_factory=list)
    name: str = "recording"
    _messages: list[dict[str, str]] | None = None

    def act(self, obs: Observation) -> Action:
        """Record prompt / injected tokens, generate the next turn, and parse it into an action."""
        if self._messages is None:
            self._messages = [
                {"role": "system", "content": system_prompt(obs.task, obs.schema)},
                {"role": "user", "content": _ACTION_INSTRUCTIONS},
            ]
            for message in self._messages:
                self.prompt_ids.extend(self.encode(message["content"]))
        elif obs.last is not None:
            result_text = obs.last.text[: self.max_tool_output_chars]
            content = (
                f"Tool {obs.last.tool} result:\n{result_text}\n"
                "Respond with the next action as JSON."
            )
            self._messages.append({"role": "user", "content": content})
            self._append_completion(self.encode(content), None)  # injected tokens: masked out

        generation = self.generate_fn(self._messages, self.max_new_tokens)
        self._messages.append({"role": "assistant", "content": generation.text})
        self._append_completion(generation.token_ids, generation.logprobs)  # policy tokens: trained
        self.parsed_flags.append(_generation_parsed(generation.text))
        return _parse_action(generation.text, self.answer_type)

    def _append_completion(self, token_ids: list[int], logprobs: list[float] | None) -> None:
        """Append a token span to the completion, masking it in (assistant) or out (injected)."""
        self.completion_ids.extend(token_ids)
        if logprobs is None:
            self.completion_mask.extend(0 for _ in token_ids)
            self.logprobs.extend(0.0 for _ in token_ids)
        else:
            if len(logprobs) != len(token_ids):
                raise ValueError("logprobs must align one-to-one with token_ids")
            self.completion_mask.extend(1 for _ in token_ids)
            self.logprobs.extend(logprobs)


class EnvRollout:
    """Generate GRPO rollouts against the environment and expose their rewards to the reward fn.

    One instance is shared between the ``rollout_func`` (:meth:`generate_batch`) and the reward
    function (:meth:`reward_fn`) so a verdict is computed exactly once, in the rollout, and read
    back by id — keeping :func:`verify` the single source of correctness.
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        encode: Encode,
        tools: ToolContext,
        rollout_config: RolloutConfig,
        verifier_config: VerifierConfig,
        reward_config: TrainingRewardConfig,
    ) -> None:
        """Bind the injected generation/tokenization, the sandbox, and the reward config."""
        self._generate_fn = generate_fn
        self._encode = encode
        self._tools = tools
        self._rollout_config = rollout_config
        self._episode_config = EpisodeConfig(
            max_turns=rollout_config.max_turns, tool_budget=rollout_config.tool_budget
        )
        self._verifier_config = verifier_config
        self._reward_config = reward_config
        self._rewards: dict[str, RewardParts] = {}

    def generate_batch(self, tasks: list[Task]) -> list[EpisodeRollout]:
        """Roll out one episode per task in ``tasks`` (the batch, already group-expanded).

        The batch is a flat list in which each task appears ``num_generations`` times; every
        occurrence becomes one independent episode with a unique id ``f"{task_id}#{k}"``. The
        per-episode reward is cached under that id for :meth:`reward_fn`.

        Args:
            tasks: The batch of tasks to roll out, in order.

        Returns:
            One :class:`EpisodeRollout` per task, in the input order.
        """
        self._rewards.clear()
        occurrence: Counter[str] = Counter()
        rollouts: list[EpisodeRollout] = []
        for task in tasks:
            k = occurrence[task.task_id]
            occurrence[task.task_id] += 1
            rollouts.append(self._rollout_one(task, f"{task.task_id}#{k}"))
        return rollouts

    def _rollout_one(self, task: Task, episode_id: str) -> EpisodeRollout:
        """Run and score a single episode, caching its reward under ``episode_id``."""
        agent = _RecordingAgent(
            generate_fn=self._generate_fn,
            encode=self._encode,
            answer_type=task.answer_type,
            max_new_tokens=self._rollout_config.max_new_tokens_per_turn,
            max_tool_output_chars=self._rollout_config.max_tool_output_chars,
        )
        trajectory = run_episode(task, agent, self._tools, self._episode_config)
        verdict = verify(
            task.expected, trajectory.final_answer, task.answer_type, self._verifier_config
        )
        all_actions_parse = (
            sum(1.0 for ok in agent.parsed_flags if ok) / len(agent.parsed_flags)
            if agent.parsed_flags
            else 1.0
        )
        final_parsed = bool(agent.parsed_flags) and agent.parsed_flags[-1]
        well_formed_final = (
            1.0 if (trajectory.stop_reason == "final_answer" and final_parsed) else 0.0
        )
        signals = signals_from_trajectory(
            trajectory,
            all_actions_parse=all_actions_parse,
            well_formed_final_answer=well_formed_final,
        )
        reward = episode_reward(verdict.correct, signals, self._reward_config)
        self._rewards[episode_id] = reward
        return EpisodeRollout(
            episode_id=episode_id,
            task_id=task.task_id,
            prompt_ids=agent.prompt_ids,
            completion_ids=agent.completion_ids,
            completion_mask=agent.completion_mask,
            logprobs=agent.logprobs,
            trajectory=trajectory,
            verdict=verdict,
            reward=reward,
        )

    def reward_fn(
        self,
        prompts: list[object] | None = None,
        completions: list[object] | None = None,
        completion_ids: list[object] | None = None,
        *,
        task_id: list[str] | None = None,
        **kwargs: object,
    ) -> list[float]:
        """The GRPO reward function: return the cached per-episode reward for each completion.

        TRL passes the batch's dataset columns as keyword args; ``task_id`` arrives in the same
        order the rollout generated, so the ``k``-th occurrence of a task here maps to the episode
        id ``f"{task_id}#{k}"`` minted in :meth:`generate_batch`. Rewards are read from the cache,
        never recomputed, so :func:`verify` runs once per episode.

        Args:
            prompts: The batch prompts (unused; rewards are keyed by task occurrence).
            completions: The batch completions (used only for a length check).
            completion_ids: The batch completion token ids (unused; see the class docstring on why
                the cache is *not* keyed by completion content).
            task_id: The batch's ``task_id`` column, in order.
            **kwargs: Other dataset columns TRL forwards.

        Returns:
            One reward float per completion, aligned to the batch order.

        Raises:
            ValueError: If ``task_id`` is missing, a length check fails, or an id is not cached.
        """
        if task_id is None:
            raise ValueError("reward_fn requires the batch 'task_id' column")
        if completions is not None and len(completions) != len(task_id):
            raise ValueError("completions and task_id must have equal length")
        occurrence: Counter[str] = Counter()
        rewards: list[float] = []
        for tid in task_id:
            k = occurrence[tid]
            occurrence[tid] += 1
            key = f"{tid}#{k}"
            if key not in self._rewards:
                raise ValueError(f"no cached reward for {key!r}: rollout/reward misaligned")
            rewards.append(self._rewards[key].reward)
        return rewards
