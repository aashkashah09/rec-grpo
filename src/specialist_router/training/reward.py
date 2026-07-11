"""The GRPO training reward: ``R = w_correct * correct + w_format * format_score``.

Pure functions only, so the training signal is trivially testable and reproducible. Two
deliberate choices (ADR-011):

* **Correctness is the deterministic verifier's verdict** (``Verdict.correct``) — the same pure
  verifier used everywhere else in the project. There is no LLM judge and no trajectory-level
  importance sampling: this is a per-episode scalar reward that GRPO turns into a group-relative
  advantage (``CLAUDE.md`` rule #2 / PROJECT_PLAN §1.2).
* **The format term is small** (``w_format`` ≪ ``w_correct``). It rewards well-formed tool use and,
  crucially, keeps a nonzero gradient when a whole GRPO group is all-correct or all-wrong (binary
  reward → ~0 advantage); it is intentionally too small to prefer a well-formatted wrong answer
  over a correct one.

This module is distinct from :mod:`specialist_router.router.reward`, which is the *routing* reward
(``quality − λ·cost − μ·latency``) — a different quantity for a different subsystem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from specialist_router.config import TrainingRewardConfig
from specialist_router.env.records import Trajectory

# The format-score components, each measured in ``[0, 1]``. The score is the mean of the components
# named in ``TrainingRewardConfig.format_components``.
FORMAT_COMPONENTS = (
    "all_actions_parse",
    "used_tool_before_answer",
    "well_formed_final_answer",
    "within_budget",
)


@dataclass(frozen=True, slots=True)
class FormatSignals:
    """Per-episode format/tool-compliance measurements, each in ``[0, 1]``.

    ``all_actions_parse`` is the fraction of the model's emitted actions that were well-formed
    (parsed to a valid tool call or a typed final answer). The rollout, which sees every raw
    generation, can measure this precisely; :func:`signals_from_trajectory` derives a proxy from a
    recorded :class:`Trajectory` alone (used by the held-out eval harness).
    """

    all_actions_parse: float
    used_tool_before_answer: float
    well_formed_final_answer: float
    within_budget: float

    def get(self, component: str) -> float:
        """Return one named component value."""
        if component not in FORMAT_COMPONENTS:
            raise ValueError(f"unknown format component: {component!r}")
        return float(getattr(self, component))


@dataclass(frozen=True, slots=True)
class RewardParts:
    """The composite training reward and the components that produced it (all logged to W&B)."""

    correct: int
    format_score: float
    reward: float


def format_score(signals: FormatSignals, components: Sequence[str]) -> float:
    """Mean of the enabled format components (each already clamped to ``[0, 1]``).

    Args:
        signals: The per-episode format measurements.
        components: Which components to average (from ``TrainingRewardConfig.format_components``).

    Returns:
        The mean component value in ``[0, 1]``.

    Raises:
        ValueError: If ``components`` is empty or names an unknown component.
    """
    if not components:
        raise ValueError("format_components must be non-empty")
    return sum(signals.get(c) for c in components) / len(components)


def episode_reward(
    correct: bool, signals: FormatSignals, config: TrainingRewardConfig
) -> RewardParts:
    """Compute the composite per-episode GRPO reward and its parts.

    Args:
        correct: The deterministic verifier's verdict for the episode's final answer.
        signals: The episode's format/tool-compliance measurements.
        config: Reward weights and which format components to average.

    Returns:
        A :class:`RewardParts` with the integer correctness, the format score, and the reward.
    """
    correct_int = int(bool(correct))
    fmt = format_score(signals, config.format_components)
    reward = config.w_correct * correct_int + config.w_format * fmt
    return RewardParts(correct=correct_int, format_score=fmt, reward=reward)


def signals_from_trajectory(
    trajectory: Trajectory,
    *,
    all_actions_parse: float | None = None,
    well_formed_final_answer: float | None = None,
) -> FormatSignals:
    """Derive :class:`FormatSignals` from a recorded episode.

    The rollout passes precise values for ``all_actions_parse`` (measured over every raw generation)
    and ``well_formed_final_answer`` (whether the final action truly parsed vs. fell back to a
    shape-valid sentinel). When those are ``None`` — e.g. the held-out eval harness only has the
    :class:`Trajectory` — proxies are derived from the trajectory:

    * ``all_actions_parse`` ← the fraction of tool calls that executed without a sandbox error,
    * ``well_formed_final_answer`` ← whether the episode ended by submitting a (non-null) answer.

    Args:
        trajectory: The recorded episode.
        all_actions_parse: Precise well-formed-action fraction, or ``None`` to use the proxy.
        well_formed_final_answer: Precise 0/1 flag, or ``None`` to use the proxy.

    Returns:
        The per-episode :class:`FormatSignals`.
    """
    answered = trajectory.stop_reason == "final_answer"
    used_tool = len(trajectory.tool_calls) >= 1

    if all_actions_parse is None:
        n_calls = len(trajectory.tool_calls)
        all_actions_parse = (
            sum(1.0 for tc in trajectory.tool_calls if tc.ok) / n_calls if n_calls else 1.0
        )
    if well_formed_final_answer is None:
        answered_ok = answered and trajectory.final_answer is not None
        well_formed_final_answer = 1.0 if answered_ok else 0.0

    return FormatSignals(
        all_actions_parse=_clamp01(all_actions_parse),
        used_tool_before_answer=1.0 if (answered and used_tool) else 0.0,
        well_formed_final_answer=_clamp01(well_formed_final_answer),
        within_budget=1.0 if answered else 0.0,
    )


def _clamp01(value: float) -> float:
    """Clamp a value into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))
