#!/usr/bin/env python
"""Encode item text, train the RQ-VAE, and write one token sequence per item."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recgrpo.semid.assign import assign_semantic_ids, semantic_id_stats
from recgrpo.semid.encoder import encode_items
from recgrpo.semid.train_rqvae import train_rqvae
from recgrpo.utils import get_logger, load_config, set_seed

logger = get_logger("semantic_ids")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rqvae.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train["seed"])

    processed_dir = Path(cfg.processed_dir)
    with open(processed_dir / "item_texts.json") as fh:
        texts = json.load(fh)

    embeddings = encode_items(
        texts,
        model_name=cfg.encoder["model_name"],
        batch_size=cfg.encoder["batch_size"],
        normalize=cfg.encoder["normalize"],
        cache=cfg.encoder["cache"],
    )

    model = train_rqvae(embeddings, cfg, log_path="results/logs/rqvae_train.jsonl")

    device = next(model.parameters()).device
    codes = model.encode_codes(torch.as_tensor(embeddings, dtype=torch.float32, device=device))
    tokens = assign_semantic_ids(codes, capacity=cfg.dedup_capacity)

    out_dir = Path(cfg.out_dir)
    np.save(out_dir / "item_codes.npy", codes)
    np.save(out_dir / "item_tokens.npy", tokens)

    stats = semantic_id_stats(codes)
    stats["suffix_tokens_used"] = int(len(np.unique(tokens[:, -1])))
    stats["encoder"] = cfg.encoder["model_name"]
    stats["config"] = args.config
    stats["seed"] = cfg.train["seed"]
    stats["finished_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    Path("results").mkdir(exist_ok=True)
    with open("results/rqvae_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info(
        "%d items -> %d unique code triples, %d items needed a suffix, deepest collision %d",
        stats["n_items"],
        stats["unique_prefixes"],
        stats["colliding_items"],
        stats["max_collision_group"],
    )
    logger.info("active codes per level: %s", stats["active_codes"])


if __name__ == "__main__":
    main()
