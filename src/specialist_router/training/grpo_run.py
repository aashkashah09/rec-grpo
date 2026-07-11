"""GRPO training entrypoint: wire TRL's GRPOTrainer to the env-coupled rollout and verifier reward.

This is the only module that touches TRL/torch/vLLM, and it does so through **lazy, function-local
imports** so the package still imports (and ``mypy``-checks) on CPU/CI without the ``training``
extra.
Nothing here runs in CI: the real path needs a GPU, and ``--dry-run`` routes to the CPU mock in
:mod:`specialist_router.training.dry_run` instead.

Design (ADR-010):

* **Seam = ``rollout_func``.** We own the multi-turn generation loop (via
  :class:`specialist_router.training.rollout.EnvRollout`), so the Phase-1/2 JSON tool protocol, the
  verifier, and the eval harness are reused byte-for-byte; Phase 4 becomes a checkpoint swap.
* **Reward = ``EnvRollout.reward_fn``**, reading per-episode rewards the rollout already computed
  (verifier correctness + small format term). ``verify`` runs once per episode and stays pure.
* **QLoRA + colocate vLLM**, all hyperparameters from ``configs/grpo.yaml`` — nothing hard-coded.

The exact TRL 1.x parameter names are pinned in ``configs/grpo.yaml`` and mirrored here; re-verify
them against the installed TRL when provisioning the GPU box (TRL's agentic surface moves fast).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from specialist_router.config import GrpoConfig, VerifierConfig, load_config, load_grpo_config
from specialist_router.env.tools import ToolContext
from specialist_router.training.data import TaskSampler, build_task_pool, split_tasks
from specialist_router.training.rollout import EnvRollout, TurnGeneration


def build_tool_context(grpo_config: GrpoConfig) -> tuple[ToolContext, VerifierConfig, str]:
    """Build the sandboxed tool context and DB for the env the tasks are grounded in.

    Returns the :class:`ToolContext`, the verifier tolerances, and the SQLite path (kept so the
    caller can clean it up). Reuses the Phase-1 database builder and read-only sandbox.
    """
    from specialist_router.env.database import build_dataset, write_sqlite_file

    env = load_config(grpo_config.data.env_config)
    dataset = build_dataset(env.db, env.seed)
    db_path = str(Path(grpo_config.checkpoint.output_dir) / "train_env.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env.tools.run_sql, env.tools.python_calc)
    return tools, env.verifier, db_path


def build_model_and_tokenizer(config: GrpoConfig) -> tuple[Any, Any]:
    """Load the 4-bit NF4 base model with a QLoRA adapter and its tokenizer (GPU only).

    All settings come from ``config.model`` / ``config.qlora``. The heavy libraries are imported
    lazily so this module stays importable without them.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, config.model.dtype)
    quant = BitsAndBytesConfig(
        load_in_4bit=config.qlora.load_in_4bit,
        bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, trust_remote_code=config.model.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=quant,
        torch_dtype=dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.qlora.use_gradient_checkpointing
    )
    lora = LoraConfig(
        r=config.qlora.r,
        lora_alpha=config.qlora.alpha,
        lora_dropout=config.qlora.dropout,
        target_modules=config.qlora.target_modules,
        bias=config.qlora.bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), tokenizer


def build_grpo_config(config: GrpoConfig) -> Any:
    """Assemble TRL's ``GRPOConfig`` from ``config.grpo`` / ``config.vllm`` (names per ADR-010)."""
    from trl import GRPOConfig as TrlGrpoConfig

    g = config.grpo
    return TrlGrpoConfig(
        output_dir=config.checkpoint.output_dir,
        num_generations=g.num_generations,
        num_iterations=g.num_iterations,
        beta=g.beta,
        epsilon=g.epsilon,
        epsilon_high=g.epsilon_high,
        loss_type=g.loss_type,
        scale_rewards=g.scale_rewards,
        temperature=g.temperature,
        top_p=g.top_p,
        learning_rate=g.learning_rate,
        lr_scheduler_type=g.lr_scheduler_type,
        warmup_ratio=g.warmup_ratio,
        weight_decay=g.weight_decay,
        max_grad_norm=g.max_grad_norm,
        per_device_train_batch_size=g.per_device_train_batch_size,
        gradient_accumulation_steps=g.gradient_accumulation_steps,
        max_steps=g.max_steps,
        max_completion_length=config.model.max_completion_len,
        mask_truncated_completions=g.mask_truncated_completions,
        save_steps=config.checkpoint.save_every_steps,
        save_total_limit=config.checkpoint.save_total_limit,
        seed=config.seed,
        use_vllm=config.vllm.enable,
        vllm_mode=config.vllm.mode,
        vllm_gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        vllm_importance_sampling_correction=config.vllm.importance_sampling_correction,
        report_to=[],  # W&B is driven by our WandbRun callback, not the Trainer's integration
    )


def build_vllm_generate_fn(config: GrpoConfig, tokenizer: Any, llm: Any) -> Any:
    """Return a ``generate_fn(messages, max_new_tokens) -> TurnGeneration`` backed by vLLM.

    The chat template renders the running conversation to a prompt; vLLM decodes one assistant turn
    with per-token logprobs. Weight-syncing the policy into the colocate vLLM engine each step is
    handled by TRL. Validate the vLLM sampling/logprob API against the pinned version.
    """
    from vllm import SamplingParams

    def generate_fn(messages: list[dict[str, str]], max_new_tokens: int) -> TurnGeneration:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        params = SamplingParams(
            temperature=config.grpo.temperature,
            top_p=config.grpo.top_p,
            max_tokens=max_new_tokens,
            logprobs=0,  # return the chosen-token logprob per step
        )
        output = llm.generate([prompt], params)[0].outputs[0]
        token_ids = list(output.token_ids)
        logprobs = [lp[tid].logprob for tid, lp in zip(token_ids, output.logprobs, strict=True)]
        return TurnGeneration(text=output.text, token_ids=token_ids, logprobs=logprobs)

    return generate_fn


def build_rollout_func(
    env_rollout: EnvRollout, sampler: TaskSampler, task_by_id: dict[str, Any]
) -> Any:
    """Adapt :meth:`EnvRollout.generate_batch` to TRL's ``rollout_func`` contract.

    TRL supplies the group-expanded batch (each task repeated ``num_generations`` times) with the
    dataset ``task_id`` column; we roll out one episode per row and return the token tensors TRL
    trains on. Verdicts are fed to the curriculum ``sampler`` here (the only place we observe them).
    """

    def rollout_func(prompts: list[str], **kwargs: Any) -> dict[str, list[Any]]:
        task_ids: list[str] = list(kwargs["task_id"])
        tasks = [task_by_id[tid] for tid in task_ids]
        rollouts = env_rollout.generate_batch(tasks)
        for roll in rollouts:
            sampler.record_outcome(roll.task_id, bool(roll.verdict.correct))
        return {
            "prompt_ids": [r.prompt_ids for r in rollouts],
            "completion_ids": [r.completion_ids for r in rollouts],
            "completion_mask": [r.completion_mask for r in rollouts],
            "logprobs": [r.logprobs for r in rollouts],
        }

    return rollout_func


def train(
    config_path: str,
    *,
    seed: int | None = None,
    resume_from: str | None = None,
    init_from_sft: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run (or dry-run) GRPO training.

    Args:
        config_path: Path to ``configs/grpo.yaml``.
        seed: Optional seed override.
        resume_from: Optional checkpoint dir to resume from (weights/optimizer/step/RNG).
        init_from_sft: Optional SFT-warmup adapter to initialise from (explicit; never automatic).
        dry_run: If set, run the CPU mock pipeline (no GPU) and return.
    """
    config = load_grpo_config(config_path, seed_override=seed)
    if dry_run:
        from specialist_router.training.dry_run import run_mock_dry_run

        run_mock_dry_run(config)
        return

    _train_on_gpu(config, resume_from=resume_from, init_from_sft=init_from_sft)


def _train_on_gpu(
    config: GrpoConfig, *, resume_from: str | None, init_from_sft: str | None
) -> None:
    """The real GPU training path (never exercised in CI; lazy heavy imports)."""
    from datasets import Dataset
    from trl import GRPOTrainer

    from specialist_router.env.episode import system_prompt
    from specialist_router.training.callbacks import SpotInterruptGuard, WandbRun

    model, tokenizer = build_model_and_tokenizer(config)
    if init_from_sft is not None:
        model.load_adapter(init_from_sft, adapter_name="sft_warmup")
        model.set_adapter("sft_warmup")

    tools, verifier_config, _db_path = build_tool_context(config)
    pool = build_task_pool(config.data.env_config, n_override=config.data.task_pool_size)
    split = split_tasks(pool, config.data.heldout_fraction, config.data.split_seed)
    sampler = TaskSampler.from_config(split.train, config.data, config.seed)
    task_by_id = {t.task_id: t for t in split.train}

    llm = None  # TRL provides the colocate vLLM engine to rollout_func in current versions;
    generate_fn = build_vllm_generate_fn(config, tokenizer, llm)
    env_rollout = EnvRollout(
        generate_fn=generate_fn,
        encode=tokenizer.encode,
        tools=tools,
        rollout_config=config.rollout,
        verifier_config=verifier_config,
        reward_config=config.reward,
    )

    schema = tools.inspect_schema()
    train_rows = [{"task_id": t.task_id, "prompt": system_prompt(t, schema)} for t in split.train]
    dataset = Dataset.from_list(train_rows)

    trainer = GRPOTrainer(
        model=model,
        args=build_grpo_config(config),
        train_dataset=dataset,
        reward_funcs=[env_rollout.reward_fn],
        rollout_func=build_rollout_func(env_rollout, sampler, task_by_id),
    )
    wandb_run = WandbRun(config.wandb, run_config={"grpo_config": config.model_dump()})
    trainer.add_callback(
        _build_eval_callback(config, generate_fn, split.heldout, tools, verifier_config, wandb_run)
    )

    # Spot-interruption safety: on SIGTERM/SIGINT flush a checkpoint before the instance dies.
    guard = SpotInterruptGuard(
        lambda: trainer.save_model(str(Path(config.checkpoint.output_dir) / "interrupt"))
    )
    if config.checkpoint.flush_on_sigterm:
        guard.install()
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    finally:
        guard.restore()
        wandb_run.finish()
        tools.close()


def _build_eval_callback(
    config: GrpoConfig,
    generate_fn: Any,
    heldout: list[Any],
    tools: ToolContext,
    verifier_config: VerifierConfig,
    wandb_run: Any,
) -> Any:
    """Build the transformers callback that runs held-out eval, tracks drift, and keeps top-K.

    Kept here (not in ``callbacks.py``) because it is the one callback that subclasses
    ``transformers.TrainerCallback``; the pure logic it drives (``FormatRateTrend``,
    ``TopKCheckpointRegistry``) lives in ``callbacks.py`` and is unit-tested without transformers.
    The held-out eval reuses the shared harness — the same verifier path as everywhere (ADR-010).
    """
    from transformers import TrainerCallback

    from specialist_router.agents.chat_agent import ChatToolAgent
    from specialist_router.config import EpisodeConfig
    from specialist_router.evaluation.harness import evaluate
    from specialist_router.serving.clients import ChatResponse
    from specialist_router.training.callbacks import FormatRateTrend, TopKCheckpointRegistry
    from specialist_router.training.data import heldout_eval_sample

    episode_config = EpisodeConfig(
        max_turns=config.rollout.max_turns, tool_budget=config.rollout.tool_budget
    )
    eval_tasks = heldout_eval_sample(
        heldout, config.evaluation.n_heldout_tasks, config.data.split_seed
    )
    registry = TopKCheckpointRegistry(config.evaluation.keep_top_k)
    trend = FormatRateTrend()

    def _eval_agent_factory() -> ChatToolAgent:
        """A fresh eval agent using the trained policy via the same vLLM generation path."""

        def complete(messages: list[dict[str, str]], max_tokens: int) -> ChatResponse:
            gen = generate_fn(messages, max_tokens)
            return ChatResponse(
                text=gen.text, prompt_tokens=0, completion_tokens=len(gen.token_ids)
            )

        return ChatToolAgent(name="specialist", complete_fn=complete)

    class _EvalCallback(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step % config.evaluation.eval_every_steps != 0:
                return
            report = evaluate(
                _eval_agent_factory, eval_tasks, tools, episode_config, verifier_config
            )
            wandb_run.log_eval(report, step=state.global_step)
            warning = trend.record(state.global_step, report.overall_success, report.format_rate)
            if warning is not None:
                print(f"[format-drift] {warning}")
            ckpt = str(Path(config.checkpoint.output_dir) / f"heldout-step{state.global_step}")
            registry.consider(state.global_step, report.overall_success, ckpt)

    return _EvalCallback()
