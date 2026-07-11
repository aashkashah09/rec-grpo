"""CLI: run the CPU dry-run of the GRPO pipeline (mocked generation; no GPU, no heavy deps).

Validates the full wiring — env rollout -> verify -> reward -> group advantage -> trainer contract —
before any GPU is rented. See training/dry_run.py and ADR-012.
"""

from __future__ import annotations

import argparse

from specialist_router.config import load_grpo_config
from specialist_router.training.dry_run import run_mock_dry_run


def main() -> None:
    """Parse arguments, run the mocked dry-run, and print its summary."""
    parser = argparse.ArgumentParser(description="CPU dry-run of the GRPO training pipeline.")
    parser.add_argument("--config", required=True, help="Path to configs/grpo.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    args = parser.parse_args()

    config = load_grpo_config(args.config, seed_override=args.seed)
    summary = run_mock_dry_run(config)
    print(summary.render())


if __name__ == "__main__":
    main()
