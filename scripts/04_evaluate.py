#!/usr/bin/env python
"""Score checkpoints on the test split and write results/main_results.json.

Slates are cached alongside the metrics so the composition analysis does not
have to decode the test set a second time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recgrpo.eval.evaluate import score_slates, generate_slates
from recgrpo.model.transformer import SemanticIDTransformer, count_parameters
from recgrpo.utils import get_logger, load_config, set_seed
from recgrpo.workspace import load_workspace

logger = get_logger("evaluate")

CHECKPOINTS = [
    ("sft", "checkpoints/sft/best.pt"),
    ("grpo-lambda0", "checkpoints/grpo/lambda0/policy.pt"),
    ("grpo-lambda0.25", "checkpoints/grpo/lambda0.25/policy.pt"),
    ("grpo-lambda0.5", "checkpoints/grpo/lambda0.5/policy.pt"),
    ("grpo-lambda1", "checkpoints/grpo/lambda1/policy.pt"),
    ("grpo-lambda2", "checkpoints/grpo/lambda2/policy.pt"),
    ("grpo-lambda4", "checkpoints/grpo/lambda4/policy.pt"),
    ("grpo-lambda8", "checkpoints/grpo/lambda8/policy.pt"),
    # bonus-cap ablation, held at the selected lambda
    ("grpo-cap1", "checkpoints/grpo/cap1/policy.pt"),
    ("grpo-cap5", "checkpoints/grpo/cap5/policy.pt"),
    ("grpo-capinf", "checkpoints/grpo/capinf/policy.pt"),
]


def load_policy(path: str, model_cfg: dict, device: str) -> SemanticIDTransformer:
    state = torch.load(path, map_location=device)
    model = SemanticIDTransformer(**state.get("config", model_cfg)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--all", action="store_true", help="score every checkpoint listed")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ws = load_workspace(cfg.processed_dir)
    examples = ws.eval_examples(args.split, cfg.data["max_history_items"])
    logger.info("%s split: %d targets, %d of them cold", args.split, len(examples),
                int(ws.cold_mask[[e.target for e in examples]].sum()))

    visible_path = Path("checkpoints/grpo/reward_visible.npy")
    extra_slices = {}
    if visible_path.exists():
        visible = np.load(visible_path)
        extra_slices = {"cold_reward_visible": visible, "cold_never_rewarded": ws.cold_mask & ~visible}

    targets = [(args.name or Path(args.checkpoint).parent.name, args.checkpoint)] if args.checkpoint else []
    if args.all:
        targets = [(name, path) for name, path in CHECKPOINTS if Path(path).exists()]

    results = {}
    slate_dir = Path("results/slates")
    slate_dir.mkdir(parents=True, exist_ok=True)

    for name, path in targets:
        model = load_policy(path, dict(cfg.model), device)
        start = time.time()
        slates = generate_slates(
            model,
            examples,
            ws.item_tokens,
            ws.trie,
            beam_size=cfg.eval["beam_size"],
            slate_size=max(cfg.eval["topk"]),
            device=device,
            desc=name,
        )
        metrics = score_slates(
            slates, examples, ws.cold_mask, ks=tuple(cfg.eval["topk"]), extra_slices=extra_slices
        )
        results[name] = {
            "checkpoint": path,
            "config": args.config,
            "split": args.split,
            "seed": cfg.train["seed"],
            "n_parameters": count_parameters(model),
            "beam_size": cfg.eval["beam_size"],
            "eval_seconds": round(time.time() - start, 1),
            "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": metrics,
        }
        with open(slate_dir / f"{name}_{args.split}.pkl", "wb") as fh:
            pickle.dump(slates, fh)
        logger.info(
            "%-16s recall@10 %.4f  ndcg@10 %.4f  cold recall@10 %.4f",
            name,
            metrics["all"]["recall@10"],
            metrics["all"]["ndcg@10"],
            metrics.get("cold", {}).get("recall@10", 0.0),
        )

    out_path = Path("results") / ("main_results.json" if args.split == "test" else "val_results.json")
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
