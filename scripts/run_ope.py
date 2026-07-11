"""CLI: run IPS/SNIPS/DR/direct-method (with bootstrap CIs) over a logged decisions file.

Reads a propensity-logged JSONL decisions file, fits and off-policy-evaluates every candidate
policy, and writes the results JSON and Markdown table. Applies ONLY to the single routing
decision (``CONVENTIONS.md`` rule #2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specialist_router.analysis import report_tables
from specialist_router.analysis.pipeline import run_ope
from specialist_router.config import load_ope_config, load_router_config
from specialist_router.router.logger import read_decisions


def main() -> None:
    """Parse arguments, evaluate policies off-policy, and write results + table."""
    parser = argparse.ArgumentParser(description="Off-policy-evaluate router policies from a log.")
    parser.add_argument("--decisions", default="build/phase2/decisions.jsonl")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--ope-config", default="configs/ope.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="build/phase2/ope_results.json")
    args = parser.parse_args()

    router = load_router_config(args.router_config, seed_override=args.seed)
    ope = load_ope_config(args.ope_config, seed_override=args.seed)
    decisions = read_decisions(args.decisions)

    result = run_ope(decisions, router, ope)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.rows, indent=2) + "\n")
    print(report_tables.ope_table(result.rows))
    print(f"\nWrote {out} ({len(result.rows)} policies, n={len(decisions)} decisions).")


if __name__ == "__main__":
    main()
