"""CLI: dry-run the GRPO pipeline on CPU (no GPU).

Two modes:
* ``--real`` (default for ``make grpo-dryrun``): a real, tiny GRPOTrainer step on CPU using the
  ``training-cpu`` extra (torch/trl/peft/transformers; no vLLM/CUDA). Validates the actual trainer
  wiring on macOS before renting a GPU. See ADR-014.
* ``--mock`` (dependency-free): scripted generation through the real env/verifier/reward and a mock
  trainer-input-contract check. Runs anywhere, including CI (also covered by pytest).
"""

from __future__ import annotations

import argparse

from specialist_router.config import load_grpo_config


def main() -> None:
    """Parse arguments and run the selected dry-run mode."""
    parser = argparse.ArgumentParser(description="CPU dry-run of the GRPO training pipeline.")
    parser.add_argument("--config", required=True, help="Path to configs/grpo.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--real",
        dest="real",
        action="store_true",
        help="Real tiny GRPOTrainer step on CPU (needs the 'training-cpu' extra). [default]",
    )
    mode.add_argument(
        "--mock",
        dest="real",
        action="store_false",
        help="Dependency-free mock pipeline (no torch/trl).",
    )
    parser.set_defaults(real=True)
    args = parser.parse_args()

    config = load_grpo_config(args.config, seed_override=args.seed)
    if args.real:
        from specialist_router.training.grpo_run import dry_run_cpu

        dry_run_cpu(config)
    else:
        from specialist_router.training.dry_run import run_mock_dry_run

        print(run_mock_dry_run(config).render())


if __name__ == "__main__":
    main()
