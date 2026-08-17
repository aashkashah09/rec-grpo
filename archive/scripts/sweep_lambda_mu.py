"""CLI: λ/μ reward-weight sensitivity sweep over the existing logged decisions.

Re-scores the logged decisions across a (λ, μ) grid (no new traffic), refits the learned policies
under each reward, off-policy-evaluates by DR, and writes the sweep table + heatmap plus a summary
of the regime where a learned router beats always_api on reward. Stub/CPU-simulator data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specialist_router.analysis import report_tables
from specialist_router.analysis.pipeline import generate_traffic, run_lambda_mu_sweep
from specialist_router.config import (
    load_config,
    load_ope_config,
    load_router_config,
    load_serving_config,
)
from specialist_router.router.logger import DecisionLogger, read_decisions

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
MUS = [0.0, 0.05, 0.1, 0.2]


def _regime_summary(rows: list[dict[str, object]]) -> str:
    """A short prose summary of where the learned router beats always_api on reward."""
    wins = [r for r in rows if r["learned_beats_api"]]
    if not wins:
        return "No (λ, μ) cell in the grid has a learned router beating always_api on DR reward."
    min_lambda = min(float(r["lambda"]) for r in wins)
    best = max(wins, key=lambda r: float(r["margin_vs_api"]))
    lines = [
        f"Learned router beats always_api in {len(wins)}/{len(rows)} grid cells.",
        f"The win regime starts at λ ≥ {min_lambda:.2f} (cost weight); higher λ widens the margin.",
        (
            f"Largest margin: {float(best['margin_vs_api']):+.3f} at "
            f"λ={float(best['lambda']):.2f}, μ={float(best['mu']):.2f} "
            f"(winner: {best['best_learned_policy']})."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the λ/μ sweep and write the table, heatmap, JSON, and regime summary."""
    parser = argparse.ArgumentParser(
        description="λ/μ reward sensitivity sweep on logged decisions."
    )
    parser.add_argument("--env-config", default="configs/env.mini.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--ope-config", default="configs/ope.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--decisions", default="build/phase2/decisions.jsonl")
    parser.add_argument("--n", type=int, default=2000, help="Traffic size if the log is absent.")
    parser.add_argument("--out-dir", default="artifacts/phase2")
    args = parser.parse_args()

    env = load_config(args.env_config)
    router = load_router_config(args.router_config)
    ope = load_ope_config(args.ope_config)
    serving = load_serving_config(args.serving_config)

    decisions_path = Path(args.decisions)
    if decisions_path.exists():
        decisions = read_decisions(decisions_path)
        print(
            f"[sweep] re-scoring {len(decisions)} existing logged decisions from {decisions_path}"
        )
    else:
        decisions = generate_traffic(env, router, serving, args.n, seed=router.seed)
        with DecisionLogger(decisions_path) as logger:
            for decision in decisions:
                logger.write(decision)
        print(f"[sweep] regenerated {len(decisions)} decisions -> {decisions_path}")

    rows = run_lambda_mu_sweep(decisions, router, ope, LAMBDAS, MUS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lambda_mu_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = _regime_summary(rows)
    (out_dir / "lambda_mu_sweep_table.md").write_text(
        report_tables.lambda_mu_sweep_table(rows) + "\n\n" + summary + "\n"
    )

    try:
        from specialist_router.analysis import plots

        plots.plot_lambda_mu_heatmap(rows, LAMBDAS, MUS, out_dir / "lambda_mu_sweep.png")
        print("[sweep] wrote lambda_mu_sweep.png")
    except ImportError:
        print("[sweep] matplotlib not installed (analysis extra) — skipping heatmap")

    print(report_tables.lambda_mu_sweep_table(rows))
    print("\n" + summary)


if __name__ == "__main__":
    main()
