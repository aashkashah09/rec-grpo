"""The single typed configuration module for the project.

`CLAUDE.md` mandates that no experiment constant lives in code — everything flows through
``configs/*.yaml`` loaded here into validated, typed models. Every run takes ``--config``
and ``--seed``; ``load_config`` centralises that contract (and the seed override) so no other
module ever parses YAML or reaches for a raw dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

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


# --------------------------------------------------------------------------------------------
# Phase 2 configs: router, OPE, and serving. Each lives in its own ``configs/*.yaml`` and is a
# separate top-level model (rather than more keys on env ``Config``) so a router/OPE run never
# has to carry the data-generation knobs, and vice versa.
# --------------------------------------------------------------------------------------------


class RewardConfig(_Strict):
    """Weights and reference scales for ``reward = quality − λ·cost_norm − μ·latency_norm``.

    Normalization uses *fixed* reference scales (not dataset min/max) so the reward is stationary
    across logs and the train/replay split cannot leak — see ADR-007. λ/μ are interpretable as
    the maximum penalty (in quality units) a fully saturated cost/latency can impose.
    """

    lambda_cost: float = Field(ge=0.0)
    mu_latency: float = Field(ge=0.0)
    cost_ref_usd: float = Field(gt=0.0)
    latency_ref_s: float = Field(gt=0.0)


class FeaturesConfig(_Strict):
    """Context-featurizer settings (see ``router.features``)."""

    embedding: Literal["hashing", "minilm"] = "hashing"
    """``hashing`` (deterministic, numpy-only; used in CI) or ``minilm`` (sentence-transformers)."""

    embed_dim: int = Field(gt=0)
    ngram_min: int = Field(gt=0)
    ngram_max: int = Field(gt=0)
    len_norm_chars: float = Field(gt=0.0)
    len_norm_tokens: float = Field(gt=0.0)
    count_norm: float = Field(gt=0.0)


class EpsilonGreedyConfig(_Strict):
    """Epsilon-greedy policy: ridge value model per arm, uniform ε exploration."""

    epsilon: float = Field(ge=0.0, le=1.0)
    ridge_lambda: float = Field(gt=0.0)


class LinUCBConfig(_Strict):
    """LinUCB policy: disjoint per-arm ridge with a UCB bonus, softmaxed for logged propensity."""

    alpha: float = Field(ge=0.0)
    ridge_lambda: float = Field(gt=0.0)
    temperature: float = Field(gt=0.0)


class ThompsonConfig(_Strict):
    """Thompson sampling over per-arm Bayesian logistic models of the (binary) quality reward."""

    prior_variance: float = Field(gt=0.0)
    newton_steps: int = Field(gt=0)
    n_posterior_samples: int = Field(gt=0)


class PoliciesConfig(_Strict):
    """Hyperparameters for every target policy the router can propose."""

    epsilon_greedy: EpsilonGreedyConfig
    linucb: LinUCBConfig
    thompson: ThompsonConfig


class RouterConfig(_Strict):
    """Top-level router configuration (reward, featurizer, policies, logging policy)."""

    schema_version: int
    seed: int
    logging_policy: Literal["uniform"] = "uniform"
    reward: RewardConfig
    features: FeaturesConfig
    policies: PoliciesConfig


class DrConfig(_Strict):
    """Doubly-robust settings: the reward (outcome) model and its cross-fitting."""

    reward_model: Literal["ridge", "gbm"] = "ridge"
    ridge_lambda: float = Field(gt=0.0)
    n_folds: int = Field(ge=2)


class OpeConfig(_Strict):
    """Off-policy-evaluation settings (bootstrap CIs, weight clipping, DR outcome model)."""

    schema_version: int
    seed: int
    n_bootstrap: int = Field(gt=0)
    ci_alpha: float = Field(gt=0.0, lt=1.0)
    weight_clip: float | None = Field(default=None)
    dr: DrConfig


class StubArmConfig(_Strict):
    """Generative quality/cost/latency profile for one simulated arm (CPU dev, no GPU/API).

    Quality is a Bernoulli whose success probability is ``base_quality`` minus a per-difficulty
    penalty (the simulator, unlike the router, may see the true difficulty). Cost and latency are
    clipped Gaussians. All draws are seeded per decision, so traffic is reproducible.
    """

    base_quality: float = Field(ge=0.0, le=1.0)
    difficulty_penalty: float = Field(ge=0.0)
    cost_usd_mean: float = Field(ge=0.0)
    cost_usd_std: float = Field(ge=0.0)
    latency_s_mean: float = Field(gt=0.0)
    latency_s_std: float = Field(ge=0.0)


class StubConfig(_Strict):
    """The two simulated arms used for CPU traffic generation and ``repro-phase2``."""

    local: StubArmConfig
    api: StubArmConfig


class EndpointConfig(_Strict):
    """A real chat endpoint for a live agent (OpenAI-compatible; used only outside CI)."""

    base_url: str
    model: str
    api_key_env: str | None = None
    price_prompt_per_1k_usd: float = Field(ge=0.0)
    price_completion_per_1k_usd: float = Field(ge=0.0)
    timeout_s: float = Field(gt=0.0)


class ServingConfig(_Strict):
    """Serving/agent configuration: which backend, the stub profiles, and real endpoints."""

    schema_version: int
    seed: int
    backend: Literal["stub", "real"] = "stub"
    stub: StubConfig
    local_endpoint: EndpointConfig | None = None
    api_endpoint: EndpointConfig | None = None


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_yaml_model(
    path: str | Path, model: type[_ModelT], seed_override: int | None = None
) -> _ModelT:
    """Load and validate a YAML file into ``model``, optionally overriding a ``seed`` field.

    Args:
        path: Path to a ``configs/*.yaml`` file.
        model: The pydantic model class to validate against.
        seed_override: If given and the model has a ``seed`` field, replaces it.

    Returns:
        A validated instance of ``model``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the file is missing keys or has the wrong types.
    """
    raw = yaml.safe_load(Path(path).read_text())
    obj = model.model_validate(raw)
    if seed_override is not None and "seed" in model.model_fields:
        obj = obj.model_copy(update={"seed": seed_override})
    return obj


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
    return _load_yaml_model(path, Config, seed_override)


def load_router_config(path: str | Path, seed_override: int | None = None) -> RouterConfig:
    """Load and validate a router config (``configs/router.yaml``)."""
    return _load_yaml_model(path, RouterConfig, seed_override)


def load_ope_config(path: str | Path, seed_override: int | None = None) -> OpeConfig:
    """Load and validate an OPE config (``configs/ope.yaml``)."""
    return _load_yaml_model(path, OpeConfig, seed_override)


def load_serving_config(path: str | Path, seed_override: int | None = None) -> ServingConfig:
    """Load and validate a serving config (``configs/serving.yaml``)."""
    return _load_yaml_model(path, ServingConfig, seed_override)
