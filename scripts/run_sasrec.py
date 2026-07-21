#!/usr/bin/env python
"""SASRec baseline via RecBole, scored on the same split and the same cold slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recgrpo.train.sasrec_baseline import cold_slice_from_ranked_lists, run
from recgrpo.utils import get_logger, load_config
from recgrpo.workspace import load_workspace

logger = get_logger("sasrec")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sasrec.yaml")
    parser.add_argument("--ranked-lists", default="data/recbole/sasrec_topk.npy",
                        help="top-k item ids per test row, dumped from the trained model")
    args = parser.parse_args()

    cfg = load_config("configs/sft.yaml")
    ws = load_workspace(cfg.processed_dir)

    result = run(args.config)
    logger.info("recbole test metrics: %s", result["test_result"])

    payload = {
        "model": "sasrec",
        "framework": "recbole",
        "config": args.config,
        "seed": load_config(args.config).get("seed", 42),
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metrics": {"all": {k.lower(): v for k, v in result["test_result"].items()}},
    }

    ranked_path = Path(args.ranked_lists)
    if ranked_path.exists():
        ranked = np.load(ranked_path)
        targets = ws.split.test["item"].to_numpy()
        payload["metrics"]["cold"] = cold_slice_from_ranked_lists(ranked, targets, ws.cold_mask)
        logger.info("cold slice: %s", payload["metrics"]["cold"])

    out = Path("results/sasrec_results.json")
    out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
