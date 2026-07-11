"""The single typed configuration module for the project.

`CLAUDE.md` mandates that no experiment constant lives in code — everything flows through
``configs/*.yaml`` loaded here into validated, typed models. Every run takes ``--config``
and ``--seed``; ``load_config`` centralises that contract (and the seed override) so no other
module ever parses YAML or reaches for a raw dict.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base for config models: reject unknown keys so typos fail loudly, not silently."""

    model_config = ConfigDict(extra="forbid")


class DbConfig(_Strict):
    """Row counts, date window, vocabularies, and event probabilities for data generation."""

    n_customers: int = Field(gt=0)
    n_products: int = Field(gt=0)
    n_orders: int = Field(gt=0)
    date_start: str
    date_end: str
    segments: list[str] = Field(min_length=1)
    countries: list[str] = Field(min_length=1)
    channels: list[str] = Field(min_length=1)
    marketing_channels: list[str] = Field(min_length=1)
    categories: list[str] = Field(min_length=1)
    price_cents_min: int = Field(gt=0)
    price_cents_max: int = Field(gt=0)
    cost_fraction: float = Field(gt=0.0, lt=1.0)
    max_items_per_order: int = Field(gt=0)
    max_quantity: int = Field(gt=0)
    cancel_prob: float = Field(ge=0.0, le=1.0)
    pending_prob: float = Field(ge=0.0, le=1.0)
    refund_prob: float = Field(ge=0.0, le=1.0)
    return_prob: float = Field(ge=0.0, le=1.0)
    discount_prob: float = Field(ge=0.0, le=1.0)
    null_discount_prob: float = Field(ge=0.0, le=1.0)


class TasksConfig(_Strict):
    """Which templates to sample, how many tasks, and per-template parameter ranges."""

    n_tasks: int = Field(gt=0)
    templates: list[str] = Field(min_length=1)
    k_choices: list[int] = Field(min_length=1)
    min_units_choices: list[int] = Field(min_length=1)


class RunSqlConfig(_Strict):
    """Sandbox limits for the ``run_sql`` tool."""

    max_rows: int = Field(gt=0)
    max_ops: int = Field(gt=0)


class PythonCalcConfig(_Strict):
    """Sandbox limits for the ``python_calc`` ast-whitelist evaluator."""

    max_nodes: int = Field(gt=0)
    max_exponent: int = Field(gt=0)


class ToolsConfig(_Strict):
    """Tool-layer sandbox configuration."""

    run_sql: RunSqlConfig
    python_calc: PythonCalcConfig


class EpisodeConfig(_Strict):
    """Episode-loop budgets."""

    max_turns: int = Field(gt=0)
    tool_budget: int = Field(gt=0)


class VerifierConfig(_Strict):
    """Numeric tolerances, per answer type (see ``env.verifier``)."""

    money_abs_usd: float = Field(gt=0.0)
    ratio_rel: float = Field(gt=0.0)
    ratio_abs: float = Field(gt=0.0)
    pp_rel: float = Field(gt=0.0)
    pp_abs: float = Field(gt=0.0)


class Config(_Strict):
    """Top-level typed configuration for one environment run."""

    schema_version: int
    seed: int
    db: DbConfig
    tasks: TasksConfig
    tools: ToolsConfig
    episode: EpisodeConfig
    verifier: VerifierConfig


def load_config(path: str | Path, seed_override: int | None = None) -> Config:
    """Load and validate a YAML config, optionally overriding the seed.

    The seed override exists because every CLI entrypoint accepts ``--seed`` and must be able
    to re-run the same config under a different seed without editing the file.

    Args:
        path: Path to a ``configs/*.yaml`` file.
        seed_override: If given, replaces ``seed`` from the file.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the file is missing keys or has the wrong types.
    """
    raw = yaml.safe_load(Path(path).read_text())
    config = Config.model_validate(raw)
    if seed_override is not None:
        config = config.model_copy(update={"seed": seed_override})
    return config
