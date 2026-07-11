"""The single typed configuration module for the project.

`CLAUDE.md` mandates that no experiment constant lives in code — everything flows through
``configs/*.yaml`` loaded here into validated, typed models. Every run takes ``--config``
and ``--seed``; ``load_config`` centralises that contract (and the seed override) so no other
module ever parses YAML or reaches for a raw dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


# --------------------------------------------------------------------------------------------
# Phase 3 configs: GRPO specialist training (``configs/grpo.yaml``). This module stays pure typed
# config — the heavy training deps (torch/trl/peft/vllm/wandb) are NEVER imported here, so the
# config (and the CPU dry-run) load and validate on CI without a GPU. Parameter names mirror the
# TRL 1.x ``GRPOConfig`` surface confirmed in ADR-010; re-verify them against the pinned TRL when
# the GPU box is provisioned.
# --------------------------------------------------------------------------------------------


class ModelConfig(_Strict):
    """The base model to fine-tune, plus sequence and generation length limits.

    ``max_completion_len`` caps the *total* tokens across the whole multi-turn conversation (all
    assistant turns + injected tool results), which is how TRL 1.x interprets it — not per-turn.
    """

    name: str
    max_seq_len: int = Field(gt=0)
    max_prompt_len: int = Field(gt=0)
    max_completion_len: int = Field(gt=0)
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class QloraConfig(_Strict):
    """4-bit NF4 QLoRA adapter configuration (``peft.LoraConfig`` + bitsandbytes)."""

    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    target_modules: list[str] = Field(min_length=1)
    bias: Literal["none", "all", "lora_only"] = "none"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    use_gradient_checkpointing: bool = True


class VllmConfig(_Strict):
    """Colocated vLLM generation settings for GRPO rollouts.

    ``importance_sampling_correction`` stays on for colocate mode (correcting the
    generation/training policy mismatch); turning it off is a known GRPO footgun (ADR-010).
    """

    enable: bool = True
    mode: Literal["colocate", "server"] = "colocate"
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    enable_sleep_mode: bool = True
    tensor_parallel_size: int = Field(gt=0)
    importance_sampling_correction: bool = True


class GrpoTrainerConfig(_Strict):
    """TRL ``GRPOConfig`` hyperparameters (names mirror TRL 1.x; see ADR-010).

    ``num_generations`` is the group size G; it must be ``> 1`` or the group-relative advantage is
    undefined. The effective rollout batch (``per_device_train_batch_size *
    gradient_accumulation_steps``) must be divisible by ``num_generations`` so every group's G
    completions stay together on-device.
    """

    num_generations: int = Field(gt=1)
    num_iterations: int = Field(gt=0)
    beta: float = Field(ge=0.0)
    epsilon: float = Field(gt=0.0)
    epsilon_high: float | None = Field(default=None)
    loss_type: Literal["grpo", "dapo", "bnpo"] = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    learning_rate: float = Field(gt=0.0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)
    max_grad_norm: float = Field(gt=0.0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    mask_truncated_completions: bool = True

    @model_validator(mode="after")
    def _batch_divisible_by_group(self) -> Self:
        """The rollout batch must partition into whole groups of ``num_generations``."""
        rollout_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if rollout_batch % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size * gradient_accumulation_steps ({rollout_batch}) "
                f"must be divisible by num_generations ({self.num_generations})"
            )
        return self


class RolloutConfig(_Strict):
    """Env-coupled rollout budgets and tool-output caps.

    The caps bound token/KV growth: multi-turn SQL episodes can emit large result tables, and every
    tool-result token feeds back into the model's context and the colocate KV cache.
    """

    max_turns: int = Field(gt=0)
    tool_budget: int = Field(gt=0)
    max_tool_output_chars: int = Field(gt=0)
    max_new_tokens_per_turn: int = Field(gt=0)


class TrainingRewardConfig(_Strict):
    """Weights for the GRPO training reward ``R = w_correct*correct + w_format*format_score``.

    ``format_score`` is the mean of the enabled components, each in ``[0, 1]``. The correctness
    term (from the deterministic verifier) dominates; the small format term breaks ties and keeps
    gradients alive against binary-reward group collapse (ADR-011).
    """

    w_correct: float = Field(ge=0.0)
    w_format: float = Field(ge=0.0)
    format_components: list[
        Literal[
            "all_actions_parse",
            "used_tool_before_answer",
            "well_formed_final_answer",
            "within_budget",
        ]
    ] = Field(min_length=1)


class DataSplitConfig(_Strict):
    """Task-pool generation and the deterministic template-balanced train/held-out split.

    ``curriculum_min/max_pass_rate`` define an optional band: once a task has enough observed
    rollouts, the sampler down-weights tasks the model always fails or always solves (both give
    ~0 group-relative advantage). The band is applied at sampling time, never to the held-out set.
    """

    env_config: str
    task_pool_size: int = Field(gt=0)
    heldout_fraction: float = Field(gt=0.0, lt=1.0)
    split_seed: int
    curriculum_min_pass_rate: float = Field(ge=0.0, le=1.0)
    curriculum_max_pass_rate: float = Field(ge=0.0, le=1.0)
    curriculum_min_observations: int = Field(gt=0)

    @model_validator(mode="after")
    def _band_ordered(self) -> Self:
        """The curriculum band must be a valid interval."""
        if self.curriculum_min_pass_rate > self.curriculum_max_pass_rate:
            raise ValueError(
                f"curriculum_min_pass_rate ({self.curriculum_min_pass_rate}) must be <= "
                f"curriculum_max_pass_rate ({self.curriculum_max_pass_rate})"
            )
        return self


class EvalCadenceConfig(_Strict):
    """Held-out evaluation cadence and top-K checkpoint retention."""

    eval_every_steps: int = Field(gt=0)
    n_heldout_tasks: int = Field(gt=0)
    keep_top_k: int = Field(gt=0)
    metric: Literal["overall_success"] = "overall_success"


class WandbConfig(_Strict):
    """Weights & Biases logging (disabled by default in the dry-run)."""

    mode: Literal["online", "offline", "disabled"] = "online"
    project: str
    entity: str | None = Field(default=None)
    run_name: str | None = Field(default=None)
    log_every_steps: int = Field(gt=0)


class CheckpointConfig(_Strict):
    """Checkpointing, resume, and spot-interruption safety."""

    output_dir: str
    save_every_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)
    flush_on_sigterm: bool = True


class SftConfig(_Strict):
    """Optional SFT warmup: trigger, demo generation, and training — kept separate from the RL run.

    Demos are generated through the Phase-2 frontier ``api_agent`` (reusing
    ``configs/serving.yaml -> api_endpoint``, so no new provider choice), filtered to
    verifier-correct episodes. The GRPO run consumes a warmup adapter only via an explicit
    ``--init-from-sft`` flag; it is never merged automatically (ADR-013).
    """

    enabled: bool = False
    trigger_compliance_threshold: float = Field(ge=0.0, le=1.0)
    probe_n_tasks: int = Field(gt=0)
    n_demos_min: int = Field(gt=0)
    n_demos_max: int = Field(gt=0)
    output_dir: str
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    max_seq_len: int = Field(gt=0)

    @model_validator(mode="after")
    def _demo_bounds_ordered(self) -> Self:
        """The demo-count band must be a valid interval."""
        if self.n_demos_min > self.n_demos_max:
            raise ValueError(
                f"n_demos_min ({self.n_demos_min}) must be <= n_demos_max ({self.n_demos_max})"
            )
        return self


class DryRunConfig(_Strict):
    """CPU dry-run overrides (no GPU).

    The default dry-run mocks generation and needs no heavy deps (runs in CI). ``tiny_model`` backs
    the opt-in ``@pytest.mark.training`` path that runs a real GRPOTrainer step on CPU.
    """

    env_config: str
    tiny_model: str
    num_generations: int = Field(gt=1)
    max_steps: int = Field(gt=0)
    n_tasks: int = Field(gt=0)


class GrpoConfig(_Strict):
    """Top-level Phase-3 GRPO training configuration (``configs/grpo.yaml``)."""

    schema_version: int
    seed: int
    model: ModelConfig
    qlora: QloraConfig
    vllm: VllmConfig
    grpo: GrpoTrainerConfig
    rollout: RolloutConfig
    reward: TrainingRewardConfig
    data: DataSplitConfig
    evaluation: EvalCadenceConfig
    wandb: WandbConfig
    checkpoint: CheckpointConfig
    sft: SftConfig
    dry_run: DryRunConfig


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


def load_grpo_config(path: str | Path, seed_override: int | None = None) -> GrpoConfig:
    """Load and validate a GRPO training config (``configs/grpo.yaml``)."""
    return _load_yaml_model(path, GrpoConfig, seed_override)
