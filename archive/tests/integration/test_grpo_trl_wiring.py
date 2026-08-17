"""Opt-in check that the GRPO config wiring matches the installed TRL surface.

Marked ``training`` (skipped by default and in CI) and additionally guarded by ``importorskip`` so
it only runs when the ``training`` extra is installed (``uv sync --extra training``). CPU-safe:
constructing a ``GRPOConfig`` needs no GPU, so this catches TRL parameter-name drift — the flagged
Phase-3 risk (ADR-010) — cheaply. Run with ``uv run pytest -m training``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_GRPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "grpo.yaml"


@pytest.mark.training
def test_build_grpo_config_matches_installed_trl() -> None:
    pytest.importorskip("trl")

    from specialist_router.config import load_grpo_config
    from specialist_router.training.grpo_run import build_grpo_config

    config = load_grpo_config(_GRPO_CONFIG)
    trl_config = build_grpo_config(config)

    # If any of these names drifted in the pinned TRL, construction above would already have raised.
    assert trl_config.num_generations == config.grpo.num_generations
    assert trl_config.beta == config.grpo.beta
    assert trl_config.num_iterations == config.grpo.num_iterations
    assert trl_config.max_completion_length == config.model.max_completion_len
