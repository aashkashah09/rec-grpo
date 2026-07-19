#!/usr/bin/env python
"""GRPO post-training, optionally sweeping the cold weight.

Rollouts, rewards and lambda selection all live in the validation window. The
test split is not touched here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recgrpo.eval.evaluate import evaluate
from recgrpo.model.transformer import SemanticIDTransformer
from recgrpo.train.grpo import GRPOTrainer, build_prompts, reward_visible_mask
from recgrpo.utils import get_logger, load_config, set_seed
from recgrpo.workspace import load_workspace

logger = get_logger("grpo")


def load_sft(path: str, device: str) -> SemanticIDTransformer:
    state = torch.load(path, map_location=device)
    model = SemanticIDTransformer(**state["config"]).to(device)
    model.load_state_dict(state["model"])
    return model


def validation_metrics(model, ws, cfg, device, subsample=None, desc="val"):
    examples = ws.eval_examples(
        "val",
        cfg.model["max_history_items"],
        subsample=subsample,
        seed=cfg.train["seed"],
    )
    metrics, _ = evaluate(
        model,
        examples,
        ws.item_tokens,
        ws.trie,
        ws.cold_mask,
        ks=(10,),
        device=device,
        desc=desc,
    )
    return {
        "recall@10": round(metrics["all"]["recall@10"], 6),
        "ndcg@10": round(metrics["all"]["ndcg@10"], 6),
        "cold_recall@10": round(metrics.get("cold", {}).get("recall@10", 0.0), 6),
    }


def run_one(cold_lambda: float, args, ws, cold_cap: int | None = None) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {"reward.cold_lambda": cold_lambda}
    if cold_cap is not None:
        overrides["reward.cold_cap"] = cold_cap
    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg.train["seed"])

    model = load_sft(cfg.sft_checkpoint, device)
    prompts = build_prompts(ws.split.train, ws.split.val, cfg.model["max_history_items"])
    logger.info(
        "lambda=%g cap=%s | %d rollout prompts",
        cold_lambda,
        cfg.reward["cold_cap"] or "none",
        len(prompts),
    )

    trainer = GRPOTrainer(model, ws.trie, ws.item_tokens, ws.cold_mask, cfg, device=device)

    def eval_fn(policy, step):
        metrics = validation_metrics(
            policy, ws, cfg, device, subsample=cfg.train["eval_subsample"], desc=f"val s{step}"
        )
        return {**metrics, "select_metric": metrics["cold_recall@10"]}

    if cold_cap is None:
        tag = f"lambda{cold_lambda:g}"
    else:
        tag = f"cap{cold_cap if cold_cap > 0 else 'inf'}"
    summary = trainer.train(prompts, log_path=f"results/logs/grpo_{tag}.jsonl", eval_fn=eval_fn)
    trainer.save(Path(cfg.checkpoint_dir) / tag / "policy.pt")

    visible = reward_visible_mask(prompts, ws.n_items, ws.cold_mask)
    np.save(Path(cfg.checkpoint_dir) / "reward_visible.npy", visible)
    logger.info("reward-visible cold items: %d of %d", int(visible.sum()), int(ws.cold_mask.sum()))

    return {
        "cold_lambda": cold_lambda,
        "cold_cap": cfg.reward["cold_cap"],
        "steps": summary["steps"],
        "checkpoint": str(Path(cfg.checkpoint_dir) / tag / "policy.pt"),
        "validation": validation_metrics(trainer.model, ws, cfg, device, desc=f"val {tag}"),
    }


def select(candidates: list[dict], sft_validation: dict, rule: dict) -> float:
    """Most cold-weighted policy that keeps accuracy at or above the SFT model."""
    floor = sft_validation[rule["constraint"]["metric"]]
    feasible = [
        c for c in candidates if c["validation"][rule["constraint"]["metric"]] >= floor
    ]
    pool = feasible or candidates
    return max(pool, key=lambda c: c["validation"][rule["objective"]])["cold_lambda"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--sweep", action="store_true", help="run every lambda in the config sweep")
    parser.add_argument(
        "--cap-ablation",
        action="store_true",
        help="hold lambda fixed and vary the bonus cap instead",
    )
    parser.add_argument("--cold-lambda", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ws = load_workspace(cfg.processed_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.cap_ablation:
        lam = cfg.ablation["cold_lambda"]
        for cap in cfg.ablation["cold_cap"]:
            run_one(lam, args, ws, cold_cap=cap)
        return

    if args.sweep:
        lambdas = cfg.sweep["cold_lambda"]
    else:
        lambdas = [args.cold_lambda if args.cold_lambda is not None else cfg.reward["cold_lambda"]]

    candidates = [run_one(lam, args, ws) for lam in lambdas]

    if args.sweep:
        sft_validation = validation_metrics(
            load_sft(cfg.sft_checkpoint, device), ws, cfg, device, desc="val sft"
        )
        selected = select(candidates, sft_validation, cfg.sweep)
        logger.info("selected lambda=%g", selected)
        Path("results").mkdir(exist_ok=True)
        with open("results/grpo_selection.json", "w") as fh:
            json.dump(
                {
                    "config": args.config,
                    "seed": cfg.train["seed"],
                    "select_on": cfg.sweep["select_on"],
                    "objective": cfg.sweep["objective"],
                    "constraint": cfg.sweep["constraint"],
                    "sft_validation": sft_validation,
                    "candidates": candidates,
                    "selected": selected,
                    "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                },
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
