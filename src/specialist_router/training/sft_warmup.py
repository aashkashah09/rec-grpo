"""Optional SFT warmup: teach tool-format compliance before GRPO — kept strictly separate from RL.

Flow (ADR-013):

1. **Probe** the base model's tool-format compliance on a small task sample (reusing the eval
   harness's format-rate). If it is at/above ``sft.trigger_compliance_threshold`` the warmup is not
   needed and we say so.
2. **Generate demos** by running the Phase-2 frontier ``api_agent`` over training-split tasks and
   keeping only the episodes the deterministic verifier marks correct (so the warmup teaches
   *correct* tool use, not merely well-formed tool use). This step **spends frontier-API money**.
3. **Train** a LoRA adapter with TRL's ``SFTTrainer`` (GPU). The adapter is consumed by GRPO only
   via an explicit ``grpo_run --init-from-sft`` — never merged automatically.

**Spend safety (required):** :func:`main` prints an estimated API cost and refuses to make any
frontier-API call unless ``--confirm-spend`` is passed. The estimate is an a-priori projection
(clearly labeled), not a measured result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specialist_router.config import (
    EndpointConfig,
    EpisodeConfig,
    GrpoConfig,
    SftConfig,
    VerifierConfig,
    load_grpo_config,
    load_serving_config,
)
from specialist_router.env.episode import run_and_verify, system_prompt
from specialist_router.env.records import SftDemo, Task
from specialist_router.env.tools import ToolContext

# A-priori per-demo token assumptions for the cost ESTIMATE only (a projection, never reported as a
# measured number). Frontier tool-use episodes on these templates run a handful of turns.
_EST_TURNS_PER_DEMO = 4
_EST_PROMPT_TOKENS_PER_TURN = 900
_EST_COMPLETION_TOKENS_PER_TURN = 120


def estimate_demo_cost(endpoint: EndpointConfig, n_attempts: int) -> float:
    """Estimate the frontier-API USD cost of generating ``n_attempts`` demo episodes.

    This is a projection from fixed token assumptions and the endpoint's per-1k prices; the true
    cost depends on the frontier model's pass rate (more attempts are needed per *kept* demo when it
    is below 1) and on actual token counts. Treat it as a lower bound on spend.

    Args:
        endpoint: The frontier endpoint (prices come from ``configs/serving.yaml``).
        n_attempts: Number of episode attempts to generate.

    Returns:
        The estimated cost in USD.
    """
    prompt_k = _EST_TURNS_PER_DEMO * _EST_PROMPT_TOKENS_PER_TURN / 1000.0
    completion_k = _EST_TURNS_PER_DEMO * _EST_COMPLETION_TOKENS_PER_TURN / 1000.0
    per_attempt = (
        prompt_k * endpoint.price_prompt_per_1k_usd
        + completion_k * endpoint.price_completion_per_1k_usd
    )
    return per_attempt * n_attempts


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The base model's measured tool-format compliance and whether warmup is indicated."""

    compliance: float
    threshold: float
    n_tasks: int

    @property
    def warmup_indicated(self) -> bool:
        """Whether compliance is below the trigger threshold (so SFT warmup is worthwhile)."""
        return self.compliance < self.threshold


def probe_compliance(
    agent_factory: Callable[[], Any],
    tasks: list[Task],
    tools: ToolContext,
    episode_config: EpisodeConfig,
    verifier_config: VerifierConfig,
    threshold: float,
) -> ProbeResult:
    """Measure the base model's tool-format compliance on ``tasks`` (reuses the eval harness).

    Uses the local (base-model) endpoint, not the frontier API — this step incurs no frontier spend.
    """
    from specialist_router.evaluation.harness import evaluate

    report = evaluate(agent_factory, tasks, tools, episode_config, verifier_config)
    return ProbeResult(compliance=report.format_rate, threshold=threshold, n_tasks=len(tasks))


def generate_demos(
    endpoint: EndpointConfig,
    tasks: list[Task],
    tools: ToolContext,
    episode_config: EpisodeConfig,
    verifier_config: VerifierConfig,
    schema: str,
    n_demos_max: int,
    seed: int,
) -> list[SftDemo]:
    """Generate verifier-correct demos via the frontier ``api_agent`` (SPENDS API money).

    Only episodes the verifier marks correct are kept. Stops once ``n_demos_max`` demos are
    collected or ``tasks`` is exhausted. Callers must gate this behind an explicit spend
    confirmation (see :func:`main`).
    """
    from specialist_router.agents.api_agent import build_api_agent

    demos: list[SftDemo] = []
    for task in tasks:
        if len(demos) >= n_demos_max:
            break
        agent = build_api_agent(endpoint)
        trajectory, verdict = run_and_verify(task, agent, tools, episode_config, verifier_config)
        final = trajectory.final_answer
        if not verdict.correct or final is None:
            continue
        messages = _episode_messages(agent, task, schema, trajectory)
        demos.append(
            SftDemo(
                demo_id=f"demo-{len(demos):06d}",
                task_id=task.task_id,
                template_id=task.template_id,
                difficulty=task.difficulty,
                messages=messages,
                final_answer=final,
                source_agent=agent.name,
                seed=seed,
            )
        )
    return demos


def _episode_messages(agent: Any, task: Task, schema: str, trajectory: Any) -> list[dict[str, str]]:
    """Reconstruct the SFT chat transcript for a completed episode.

    Prefers the agent's accumulated messages (the exact conversation) when available; otherwise
    rebuilds the system+task prompt. The transcript is what the SFT trainer tokenizes.
    """
    messages = getattr(agent, "_messages", None)
    if isinstance(messages, list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
    return [{"role": "system", "content": system_prompt(task, schema)}]


def write_demos(demos: list[SftDemo], path: str) -> None:
    """Write demos as versioned JSONL (schema v3)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for demo in demos:
            handle.write(demo.model_dump_json() + "\n")


def train_sft(config: SftConfig, model_name: str, demos_path: str, seed: int) -> None:
    """Train a LoRA warmup adapter on the demos with TRL ``SFTTrainer`` (GPU only; lazy import)."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=demos_path, split="train")
    args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        max_length=config.max_seq_len,
        assistant_only_loss=True,  # train on assistant turns only (verify name in pinned TRL)
        seed=seed,
    )
    trainer = SFTTrainer(model=model_name, args=args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(config.output_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI: probe, print the cost estimate, and (only with ``--confirm-spend``) generate demos.

    Returns a process exit code: ``0`` on success, ``2`` when spend was requested but not confirmed.
    """
    parser = argparse.ArgumentParser(description="SFT warmup: demo generation + optional training.")
    parser.add_argument("--config", required=True, help="Path to configs/grpo.yaml.")
    parser.add_argument("--serving-config", required=True, help="Path to configs/serving.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--out", default=None, help="Output JSONL path for demos.")
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="REQUIRED before any frontier-API call. Without it, only the cost estimate prints.",
    )
    args = parser.parse_args(argv)

    grpo: GrpoConfig = load_grpo_config(args.config, seed_override=args.seed)
    serving = load_serving_config(args.serving_config)
    endpoint = serving.api_endpoint
    if endpoint is None:
        parser.error("configs/serving.yaml has no api_endpoint; configure the frontier endpoint.")

    n_attempts = grpo.sft.n_demos_max
    estimate = estimate_demo_cost(endpoint, n_attempts)
    print(
        f"SFT warmup demo generation would call the frontier API '{endpoint.model}' for up to "
        f"{n_attempts} episode attempts.\n"
        f"  Estimated API cost: ~${estimate:.2f} (a-priori projection; a frontier pass rate < 1 "
        f"needs more attempts per kept demo, so treat this as a LOWER bound)."
    )
    if not args.confirm_spend:
        print(
            "No API calls made. Re-run with --confirm-spend to authorize this spend and generate "
            "demos."
        )
        return 2

    # --- confirmed: from here on we may spend on the frontier API ---
    from specialist_router.env.database import build_dataset, write_sqlite_file
    from specialist_router.training.data import build_task_pool, split_tasks

    env = load_config_for(grpo)
    dataset = build_dataset(env.db, env.seed)
    db_path = str(Path(grpo.sft.output_dir) / "sft_env.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    write_sqlite_file(dataset, db_path)
    tools = ToolContext(db_path, env.tools.run_sql, env.tools.python_calc)
    episode_config = EpisodeConfig(
        max_turns=grpo.rollout.max_turns, tool_budget=grpo.rollout.tool_budget
    )
    try:
        pool = build_task_pool(grpo.data.env_config, n_override=grpo.data.task_pool_size)
        train_tasks = split_tasks(pool, grpo.data.heldout_fraction, grpo.data.split_seed).train
        demos = generate_demos(
            endpoint,
            train_tasks,
            tools,
            episode_config,
            env.verifier,
            tools.inspect_schema(),
            grpo.sft.n_demos_max,
            grpo.seed,
        )
    finally:
        tools.close()

    out_path = args.out or str(Path(grpo.sft.output_dir) / "demos.jsonl")
    write_demos(demos, out_path)
    print(f"Wrote {len(demos)} verifier-correct demos to {out_path}.")
    return 0


def load_config_for(grpo: GrpoConfig) -> Any:
    """Load the environment config referenced by the GRPO data config."""
    from specialist_router.config import load_config

    return load_config(grpo.data.env_config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
