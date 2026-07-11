"""Unit tests for the GRPO config models and their validators."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from specialist_router.config import (
    DataSplitConfig,
    GrpoTrainerConfig,
    SftConfig,
    load_grpo_config,
)

_GRPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "grpo.yaml"


def test_committed_grpo_config_loads() -> None:
    config = load_grpo_config(_GRPO_CONFIG)
    assert config.model.name.startswith("Qwen")
    assert config.grpo.num_generations > 1
    assert config.reward.w_correct > config.reward.w_format  # correctness dominates
    assert config.dry_run.num_generations > 1


def test_seed_override(tmp_path: Path) -> None:
    config = load_grpo_config(_GRPO_CONFIG, seed_override=123)
    assert config.seed == 123


def _trainer_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        num_generations=8,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        temperature=1.0,
        top_p=1.0,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=100,
    )
    base.update(overrides)
    return base


def test_batch_must_be_divisible_by_group() -> None:
    # 8 * 3 = 24 is not divisible by num_generations 5.
    with pytest.raises(ValidationError, match="divisible by num_generations"):
        GrpoTrainerConfig(**_trainer_kwargs(num_generations=5, gradient_accumulation_steps=3))


def test_group_size_must_exceed_one() -> None:
    with pytest.raises(ValidationError):
        GrpoTrainerConfig(**_trainer_kwargs(num_generations=1))


def test_curriculum_band_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="curriculum_min_pass_rate"):
        DataSplitConfig(
            env_config="configs/env.mini.yaml",
            task_pool_size=100,
            heldout_fraction=0.1,
            split_seed=0,
            curriculum_min_pass_rate=0.9,
            curriculum_max_pass_rate=0.1,
            curriculum_min_observations=4,
        )


def test_sft_demo_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="n_demos_min"):
        SftConfig(
            trigger_compliance_threshold=0.6,
            probe_n_tasks=10,
            n_demos_min=1500,
            n_demos_max=500,
            output_dir="x",
            epochs=1,
            learning_rate=1e-4,
            max_seq_len=1024,
        )
