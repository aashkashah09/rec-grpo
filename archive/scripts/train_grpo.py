"""CLI: launch GRPO specialist training (GPU) or the CPU dry-run (``--dry-run``).

Thin wrapper over training/grpo_run.py. The real path needs a GPU and the ``training`` extra; the
dry-run path runs the mocked pipeline on CPU. See ADR-010.
"""

from __future__ import annotations

import argparse

from specialist_router.training.grpo_run import train


def main() -> None:
    """Parse arguments and dispatch to the trainer."""
    parser = argparse.ArgumentParser(description="GRPO specialist training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to configs/grpo.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument(
        "--resume-from", default=None, help="Checkpoint dir to resume from (weights/optim/step)."
    )
    parser.add_argument(
        "--init-from-sft", default=None, help="SFT warmup adapter to initialise from (explicit)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the CPU mock pipeline (no GPU) and exit."
    )
    args = parser.parse_args()

    train(
        args.config,
        seed=args.seed,
        resume_from=args.resume_from,
        init_from_sft=args.init_from_sft,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
