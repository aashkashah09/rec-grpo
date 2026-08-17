"""CLI: generate logging-policy (Uniform) traffic through the stub router and write decisions JSONL.

Thin wrapper over :func:`specialist_router.analysis.pipeline.generate_traffic`. Stub backend only
(CPU; no GPU/API) — the emitted log is stub/CPU-simulator data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from specialist_router.analysis.pipeline import generate_traffic
from specialist_router.config import (
    load_config,
    load_router_config,
    load_serving_config,
)
from specialist_router.router.logger import DecisionLogger


def main() -> None:
    """Parse arguments, generate traffic, and write the propensity-logged decisions."""
    parser = argparse.ArgumentParser(description="Generate Uniform-logging-policy traffic (stub).")
    parser.add_argument("--env-config", default="configs/env.mini.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=2000, help="Number of tasks/decisions to log.")
    parser.add_argument("--out", default="build/phase2/decisions.jsonl")
    args = parser.parse_args()

    env = load_config(args.env_config, seed_override=args.seed)
    router = load_router_config(args.router_config, seed_override=args.seed)
    serving = load_serving_config(args.serving_config, seed_override=args.seed)

    decisions = generate_traffic(env, router, serving, args.n, seed=router.seed)
    out = Path(args.out)
    with DecisionLogger(out) as logger:
        for decision in decisions:
            logger.write(decision)
    print(f"Wrote {len(decisions)} decisions to {out} (seed={router.seed}).")


if __name__ == "__main__":
    main()
