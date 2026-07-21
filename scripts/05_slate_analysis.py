#!/usr/bin/env python
"""Slate composition and the memorisation check.

Reads the cached test slates, counts cold placements split by whether the
reward ever scored the item, and bootstraps the difference between the
supervised policy and each post-trained one.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recgrpo.eval.slate_analysis import compare, composition
from recgrpo.utils import get_logger, load_config
from recgrpo.workspace import load_workspace

logger = get_logger("slate_analysis")

MODELS = [
    "sft",
    "grpo-lambda0",
    "grpo-lambda0.25",
    "grpo-lambda0.5",
    "grpo-lambda1",
    "grpo-lambda2",
    "grpo-lambda4",
    "grpo-lambda8",
    "grpo-cap1",
    "grpo-cap5",
    "grpo-capinf",
]


def load_slates(name: str, split: str) -> list[list[int]]:
    with open(Path("results/slates") / f"{name}_{split}.pkl", "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--baseline", default="sft")
    parser.add_argument("--policy", default="grpo-lambda1")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ws = load_workspace(cfg.processed_dir)
    examples = ws.eval_examples(args.split, cfg.data["max_history_items"])
    visible = np.load("checkpoints/grpo/reward_visible.npy")

    logger.info(
        "%d cold items, %d reward-visible, %d never rewarded",
        int(ws.cold_mask.sum()),
        int(visible.sum()),
        int((ws.cold_mask & ~visible).sum()),
    )

    rows = []
    slates = {}
    for name in MODELS:
        slates[name] = load_slates(name, args.split)
        row = composition(name, slates[name], examples, ws.cold_mask, visible, k=args.k)
        rows.append(row.to_dict())
        logger.info(
            "%-16s cold %.2f%%  (visible %.2f%%, never %.2f%%)  conversion %.2f%%",
            name,
            100 * row.cold_share,
            100 * row.reward_visible_share,
            100 * row.never_rewarded_share,
            100 * row.cold_conversion,
        )

    Path("results").mkdir(exist_ok=True)
    with open("results/slate_composition.json", "w") as fh:
        json.dump(
            {
                "k": args.k,
                "split": args.split,
                "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "models": rows,
            },
            fh,
            indent=2,
        )

    generalization = {
        "split": args.split,
        "k": args.k,
        "baseline": args.baseline,
        "policy": args.policy,
        "n_cold_items": int(ws.cold_mask.sum()),
        "n_reward_visible_items": int(visible.sum()),
        "n_never_rewarded_items": int((ws.cold_mask & ~visible).sum()),
        "n_replicates": args.replicates,
        "seed": args.seed,
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "placement_shift": {},
    }
    for subset in ("cold", "reward_visible", "never_rewarded"):
        generalization["placement_shift"][subset] = compare(
            slates[args.baseline],
            slates[args.policy],
            examples,
            ws.cold_mask,
            visible,
            k=args.k,
            subset=subset,
            n_replicates=args.replicates,
            seed=args.seed,
        )
        stats = generalization["placement_shift"][subset]
        logger.info(
            "%-16s %.4f -> %.4f  (%+.4f pp, 95%% CI [%+.4f, %+.4f], p=%.4g)",
            subset,
            stats["baseline_share"],
            stats["policy_share"],
            100 * stats["observed"],
            100 * stats["ci_low"],
            100 * stats["ci_high"],
            stats["p_value"],
        )

    with open("results/generalization.json", "w") as fh:
        json.dump(generalization, fh, indent=2)


if __name__ == "__main__":
    main()
