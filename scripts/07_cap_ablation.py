#!/usr/bin/env python
"""Collect the bonus-cap ablation into one table.

Lambda is held at the selected operating point and only the cap moves, so the
column that matters is what happens to warm accuracy as the cap loosens.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# cap -> the model that was trained with it; cap 2 is the operating point and is
# the same policy as the selected lambda, so it is not retrained
MODELS = [
    (1, "grpo-cap1"),
    (2, "grpo-lambda1"),
    (5, "grpo-cap5"),
    (0, "grpo-capinf"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main_results.json")
    parser.add_argument("--composition", default="results/slate_composition.json")
    parser.add_argument("--selection", default="results/grpo_selection.json")
    parser.add_argument("--out", default="results/cap_ablation.csv")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    comp = {r["model"]: r for r in json.loads(Path(args.composition).read_text())["models"]}
    cold_lambda = json.loads(Path(args.selection).read_text())["selected"]

    fields = [
        "cold_cap",
        "cold_lambda",
        "model",
        "recall@10",
        "ndcg@10",
        "cold_recall@10",
        "cold_slate_share",
        "cold_conversion",
    ]
    rows = []
    for cap, name in MODELS:
        if name not in results:
            continue
        metrics = results[name]["metrics"]
        row = {
            "cold_cap": cap if cap > 0 else "none",
            "cold_lambda": cold_lambda,
            "model": name,
            "recall@10": metrics["all"]["recall@10"],
            "ndcg@10": metrics["all"]["ndcg@10"],
            "cold_recall@10": metrics["cold"]["recall@10"],
            "cold_slate_share": comp.get(name, {}).get("cold_share"),
            "cold_conversion": comp.get(name, {}).get("cold_conversion"),
        }
        rows.append(row)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
