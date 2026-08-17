"""Unit tests for the GRPO training reward and format signals."""

from __future__ import annotations

import pytest

from specialist_router.config import TrainingRewardConfig
from specialist_router.env.records import ToolCall, Trajectory
from specialist_router.training.reward import (
    FORMAT_COMPONENTS,
    FormatSignals,
    episode_reward,
    format_score,
    signals_from_trajectory,
)


def _config(w_correct: float = 1.0, w_format: float = 0.15) -> TrainingRewardConfig:
    return TrainingRewardConfig(
        w_correct=w_correct, w_format=w_format, format_components=list(FORMAT_COMPONENTS)
    )


def _perfect_signals() -> FormatSignals:
    return FormatSignals(1.0, 1.0, 1.0, 1.0)


def test_reward_correct_dominates_format() -> None:
    cfg = _config()
    correct = episode_reward(True, _perfect_signals(), cfg)
    wrong = episode_reward(False, _perfect_signals(), cfg)
    assert correct.reward == pytest.approx(1.15)
    assert wrong.reward == pytest.approx(0.15)
    # A correct answer always beats a well-formatted wrong one (format term too small to flip it).
    assert correct.reward > wrong.reward
    assert wrong.reward < cfg.w_correct


def test_format_score_is_mean_of_selected_components() -> None:
    signals = FormatSignals(
        all_actions_parse=0.5,
        used_tool_before_answer=1.0,
        well_formed_final_answer=0.0,
        within_budget=1.0,
    )
    assert format_score(signals, ["all_actions_parse", "within_budget"]) == pytest.approx(0.75)
    assert format_score(signals, list(FORMAT_COMPONENTS)) == pytest.approx(0.625)


def test_format_score_rejects_empty_and_unknown() -> None:
    with pytest.raises(ValueError):
        format_score(_perfect_signals(), [])
    with pytest.raises(ValueError):
        FormatSignals(1.0, 1.0, 1.0, 1.0).get("nonexistent")


def test_reward_zero_weights_gives_only_correctness() -> None:
    cfg = _config(w_correct=1.0, w_format=0.0)
    assert episode_reward(True, FormatSignals(0.0, 0.0, 0.0, 0.0), cfg).reward == pytest.approx(1.0)


def test_signals_from_trajectory_precise_overrides() -> None:
    traj = Trajectory(
        task_id="t", tool_calls=[], final_answer=1.0, stop_reason="final_answer", num_turns=1
    )
    signals = signals_from_trajectory(traj, all_actions_parse=0.25, well_formed_final_answer=1.0)
    assert signals.all_actions_parse == pytest.approx(0.25)
    assert signals.well_formed_final_answer == 1.0
    assert signals.within_budget == 1.0
    # No tool used before answering -> that component is 0.
    assert signals.used_tool_before_answer == 0.0


def test_signals_from_trajectory_proxies_from_tool_calls() -> None:
    traj = Trajectory(
        task_id="t",
        tool_calls=[
            ToolCall(tool="run_sql", arguments={}, result="ok", ok=True),
            ToolCall(tool="run_sql", arguments={}, result="err", ok=False),
        ],
        final_answer=[  # a list answer
            "a"
        ],
        stop_reason="final_answer",
        num_turns=3,
    )
    signals = signals_from_trajectory(traj)
    assert signals.all_actions_parse == pytest.approx(0.5)  # one of two tool calls ok
    assert signals.used_tool_before_answer == 1.0
    assert signals.well_formed_final_answer == 1.0


def test_signals_from_trajectory_truncated_episode_is_not_within_budget() -> None:
    traj = Trajectory(
        task_id="t", tool_calls=[], final_answer=None, stop_reason="max_turns", num_turns=12
    )
    signals = signals_from_trajectory(traj)
    assert signals.within_budget == 0.0
    assert signals.well_formed_final_answer == 0.0
