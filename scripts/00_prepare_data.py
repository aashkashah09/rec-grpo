#!/usr/bin/env python
"""Download Amazon Beauty, filter, split, and write the processed tables."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recgrpo.data.amazon import (
    apply_k_core,
    build_catalog,
    build_item_text,
    download,
    load_metadata,
    load_reviews,
)
from recgrpo.data.split import cold_items, cold_target_count, temporal_split
from recgrpo.train.sasrec_baseline import write_atomic_files
from recgrpo.utils import get_logger, load_config

logger = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = download(cfg.reviews_url, raw_dir / Path(cfg.reviews_url).name)
    metadata_path = download(cfg.metadata_url, raw_dir / Path(cfg.metadata_url).name)

    df = load_reviews(reviews_path)
    if cfg.drop_duplicate_pairs:
        before = len(df)
        df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="first")
        logger.info("dropped %d repeat (user, item) pairs", before - len(df))
    if cfg.min_timestamp:
        df = df[df["timestamp"] >= cfg.min_timestamp]

    df = apply_k_core(df, cfg.k_core)
    interactions, user_map, item_map = build_catalog(df)
    n_users, n_items = len(user_map), len(item_map)
    logger.info("catalog: %d users, %d items, %d interactions", n_users, n_items, len(interactions))

    meta = load_metadata(metadata_path, keep_asins=set(item_map))
    texts = build_item_text(meta, item_map, cfg.text_fields, cfg.max_description_chars)

    split = temporal_split(
        interactions,
        val_frac=cfg.split["val_frac"],
        test_frac=cfg.split["test_frac"],
        tie_break=tuple(cfg.split["tie_break"]),
    )
    cold = cold_items(split, n_items)
    n_cold_targets = cold_target_count(split, cold)

    interactions.to_parquet(processed_dir / "interactions.parquet", index=False)
    split.train.to_parquet(processed_dir / "train.parquet", index=False)
    split.val.to_parquet(processed_dir / "val.parquet", index=False)
    split.test.to_parquet(processed_dir / "test.parquet", index=False)
    np.save(processed_dir / "cold_mask.npy", cold)
    with open(processed_dir / "item_texts.json", "w") as fh:
        json.dump(texts, fh)
    with open(processed_dir / "id_maps.pkl", "wb") as fh:
        pickle.dump({"user_map": user_map, "item_map": item_map}, fh)

    write_atomic_files(interactions, "data/recbole", dataset=cfg.dataset)

    def as_date(ts: int) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    stats = {
        "dataset": f"amazon-{cfg.dataset}-2014",
        "k_core": cfg.k_core,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(len(interactions)),
        "density": round(len(interactions) / (n_users * n_items), 8),
        "split": {
            "mode": cfg.split["mode"],
            "sizes": split.sizes,
            "train_end": {"timestamp": split.train_end, "date": as_date(split.train_end)},
            "val_end": {"timestamp": split.val_end, "date": as_date(split.val_end)},
            "test_end": {
                "timestamp": int(split.test["timestamp"].max()),
                "date": as_date(int(split.test["timestamp"].max())),
            },
        },
        "cold": {
            "n_cold_items": int(cold.sum()),
            "cold_item_frac": round(float(cold.mean()), 6),
            "n_cold_test_targets": n_cold_targets,
            "cold_test_target_frac": round(n_cold_targets / len(split.test), 6),
        },
        "items_with_metadata": int(sum(1 for asin in item_map if asin in meta)),
        "config": args.config,
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("%s", json.dumps(stats["split"]["sizes"]))
    logger.info("cold items: %d carrying %d test targets", cold.sum(), n_cold_targets)


if __name__ == "__main__":
    main()
