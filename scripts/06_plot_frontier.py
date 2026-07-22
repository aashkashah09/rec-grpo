#!/usr/bin/env python
"""Build the lambda sweep table and draw the engagement-freshness frontier."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAMBDA_RE = re.compile(r"grpo-lambda([0-9.]+)$")

INK = "#1b1b1f"
MUTED = "#8a8f98"
CURVE = "#2f6f9f"
HIGHLIGHT = "#c2562c"
SFT_COLOR = "#4a4f57"
DOMINANCE = "#dcebd8"


def undominated(points: list[tuple[float, float]]) -> set[int]:
    """Indices of points no other point beats on both axes."""
    return {
        i
        for i, (x, y) in enumerate(points)
        if not any(
            x2 >= x and y2 >= y and (x2 > x or y2 > y)
            for j, (x2, y2) in enumerate(points)
            if j != i
        )
    }


def collect(results: dict, composition: dict | None) -> tuple[dict, list[dict]]:
    shares = {}
    if composition:
        shares = {row["model"]: row["cold_share"] for row in composition["models"]}

    sft = results["sft"]["metrics"]
    sft_row = {
        "model": "sft",
        "cold_lambda": None,
        "recall@10": sft["all"]["recall@10"],
        "ndcg@10": sft["all"]["ndcg@10"],
        "cold_recall@10": sft["cold"]["recall@10"],
        "cold_slate_share": shares.get("sft"),
    }

    sweep = []
    for name, payload in results.items():
        match = LAMBDA_RE.match(name)
        if not match:
            continue
        metrics = payload["metrics"]
        sweep.append(
            {
                "model": name,
                "cold_lambda": float(match.group(1)),
                "recall@10": metrics["all"]["recall@10"],
                "ndcg@10": metrics["all"]["ndcg@10"],
                "cold_recall@10": metrics["cold"]["recall@10"],
                "cold_slate_share": shares.get(name),
            }
        )
    sweep.sort(key=lambda row: row["cold_lambda"])
    return sft_row, sweep


def write_csv(sft_row: dict, sweep: list[dict], path: Path) -> None:
    fields = ["model", "cold_lambda", "recall@10", "ndcg@10", "cold_recall@10", "cold_slate_share"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(sft_row)
        for row in sweep:
            writer.writerow(row)


def plot(sft_row: dict, sweep: list[dict], selected: float, sasrec_cold: float | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)

    xs = [row["ndcg@10"] for row in sweep]
    ys = [row["cold_recall@10"] for row in sweep]
    sx, sy = sft_row["ndcg@10"], sft_row["cold_recall@10"]
    front = undominated(list(zip(xs, ys)))

    ax.margins(x=0.13, y=0.16)
    ax.scatter(xs, ys, s=0)  # establish the data range before shading
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # anything above and to the right of the supervised model beats it on both axes
    ax.add_patch(
        plt.Rectangle((sx, sy), x1 - sx, y1 - sy, facecolor=DOMINANCE, edgecolor="none", zorder=1)
    )
    ax.text(
        sx + (x1 - sx) * 0.035,
        y1 - (y1 - y0) * 0.04,
        "better than SFT\non both axes",
        ha="left",
        va="top",
        fontsize=9,
        linespacing=1.4,
        color="#5d7a55",
        zorder=2,
    )

    # the run of points that are not dominated, then the tail past the useful range
    tail = [i for i in range(len(sweep)) if i not in front]
    order = sorted(front)
    ax.plot([xs[i] for i in order], [ys[i] for i in order], "-", color=CURVE, linewidth=1.7, zorder=3)
    for i in tail:
        nearest = min(order, key=lambda j: abs(xs[j] - xs[i]))
        ax.plot(
            [xs[nearest], xs[i]], [ys[nearest], ys[i]],
            "--", color=CURVE, linewidth=1.2, alpha=0.45, zorder=3,
        )

    ax.scatter([xs[i] for i in order], [ys[i] for i in order], s=46, color=CURVE,
               zorder=5, edgecolor="white", linewidth=1.0)
    ax.scatter([xs[i] for i in tail], [ys[i] for i in tail], s=46, facecolor="white",
               edgecolor=CURVE, linewidth=1.4, alpha=0.75, zorder=5)

    ax.scatter([sx], [sy], s=105, marker="D", color=SFT_COLOR, zorder=6,
               edgecolor="white", linewidth=1.2)
    ax.annotate("SFT\n(before GRPO)", (sx, sy), textcoords="offset points", xytext=(-12, -6),
                ha="right", va="top", fontsize=9.5, color=SFT_COLOR, fontweight="bold",
                linespacing=1.4, zorder=6)

    # the move the reward actually produces
    chosen = next(i for i, row in enumerate(sweep) if abs(row["cold_lambda"] - selected) < 1e-9)
    ax.annotate(
        "", xy=(xs[chosen], ys[chosen]), xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, linewidth=1.4,
                        shrinkA=9, shrinkB=13, alpha=0.85), zorder=4,
    )

    for i, row in enumerate(sweep):
        lam = row["cold_lambda"]
        pick = i == chosen
        if pick:
            ax.scatter([xs[i]], [ys[i]], s=165, facecolor="none", edgecolor=HIGHLIGHT,
                       linewidth=2.1, zorder=7)
        label = f"$\\lambda$={lam:g}"
        if i in tail:
            label += "\n(dominated)"
        inward = xs[i] == min(xs)
        ax.annotate(
            label, (xs[i], ys[i]), textcoords="offset points",
            xytext=(-10, 9) if inward else (9, 8),
            ha="right" if inward else "left",
            fontsize=9, linespacing=1.3,
            color=HIGHLIGHT if pick else MUTED,
            fontweight="bold" if pick else "normal", zorder=7,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("NDCG@10  (engagement)", fontsize=11, color=INK)
    ax.set_ylabel("Cold-item Recall@10  (freshness)", fontsize=11, color=INK)
    ax.set_title("Engagement-freshness frontier, Amazon Beauty test split",
                 fontsize=12.5, color=INK, pad=12, loc="left")

    ax.grid(True, linewidth=0.5, color="#e3e5e9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ccd1")
    ax.tick_params(colors=MUTED, labelsize=9.5)

    note = ("$\\lambda$ weights the capped cold-hit bonus in the GRPO reward\n"
            "$\\lambda$=1 was selected on validation, before the test split was scored")
    if sasrec_cold is not None:
        note += f"\nSASRec baseline sits off this scale at {sasrec_cold:.4f} cold Recall@10"
    ax.text(0.015, 0.035, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, linespacing=1.5, color=MUTED, zorder=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # drop the writer stamp so the figure is byte-identical across environments
    fig.savefig(path, bbox_inches="tight", facecolor="white", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main_results.json")
    parser.add_argument("--composition", default="results/slate_composition.json")
    parser.add_argument("--selection", default="results/grpo_selection.json")
    parser.add_argument("--sasrec", default="results/sasrec_results.json")
    parser.add_argument("--csv-out", default="results/lambda_sweep.csv")
    parser.add_argument("--fig-out", default="assets/frontier.png")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    composition = None
    if Path(args.composition).exists():
        composition = json.loads(Path(args.composition).read_text())

    selected = 1.0
    if Path(args.selection).exists():
        selected = json.loads(Path(args.selection).read_text())["selected"]

    sasrec_cold = None
    if Path(args.sasrec).exists():
        sasrec_cold = json.loads(Path(args.sasrec).read_text())["metrics"]["cold"]["recall@10"]

    sft_row, sweep = collect(results, composition)
    write_csv(sft_row, sweep, Path(args.csv_out))
    plot(sft_row, sweep, selected, sasrec_cold, Path(args.fig_out))
    print(f"wrote {args.csv_out} and {args.fig_out}")


if __name__ == "__main__":
    main()
