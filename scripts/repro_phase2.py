"""Reproduce the full Phase-2 pipeline on CPU: traffic → OPE → replay-A/B → estimator breakage.

Runs entirely on the stub simulator (no GPU, no API, no network) and writes the committed tables,
result JSON, and figures under ``artifacts/phase2/``. Deterministic given the configs and seed, so
``make repro-phase2`` regenerates byte-identical tables. Every artifact is labeled as
stub/CPU-simulator data — real-model numbers arrive in Phase 4.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from specialist_router.analysis import report_tables
from specialist_router.analysis.pipeline import (
    generate_traffic,
    run_breakage_study,
    run_ope,
    run_replay,
)
from specialist_router.config import (
    load_config,
    load_ope_config,
    load_router_config,
    load_serving_config,
)
from specialist_router.router.logger import DecisionLogger


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    """Run the Phase-2 pipeline and write all artifacts."""
    parser = argparse.ArgumentParser(description="Reproduce Phase 2 (router + OPE) on CPU.")
    parser.add_argument("--env-config", default="configs/env.mini.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--ope-config", default="configs/ope.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override every config seed.")
    parser.add_argument("--n-traffic", type=int, default=2000)
    parser.add_argument("--n-replay", type=int, default=1500)
    parser.add_argument("--out-dir", default="artifacts/phase2")
    parser.add_argument("--build-dir", default="build/phase2")
    args = parser.parse_args()

    env = load_config(args.env_config, seed_override=args.seed)
    router = load_router_config(args.router_config, seed_override=args.seed)
    ope = load_ope_config(args.ope_config, seed_override=args.seed)
    serving = load_serving_config(args.serving_config, seed_override=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # 1. Traffic under the Uniform logging policy.
    logged = generate_traffic(env, router, serving, args.n_traffic, seed=router.seed)
    with DecisionLogger(build_dir / "decisions.jsonl") as logger:
        for decision in logged:
            logger.write(decision)
    with DecisionLogger(out_dir / "decisions_sample.jsonl") as sample:
        for decision in logged[:50]:
            sample.write(decision)
    print(f"[traffic] logged {len(logged)} decisions under the Uniform policy")

    # 2. OPE over every candidate policy.
    ope_run = run_ope(logged, router, ope)
    _write_json(out_dir / "ope_results.json", ope_run.rows)
    (out_dir / "ope_table.md").write_text(report_tables.ope_table(ope_run.rows) + "\n")
    print("[ope] evaluated", len(ope_run.rows), "policies (IPS/SNIPS/DM/DR + CIs)")

    # 3. Replay-A/B validation (reward-scale parity enforced in ope.replay.calibrate).
    replay = run_replay(
        env, router, serving, ope_run, logged, ope, args.n_replay, seed=router.seed + 1
    )
    calibration_rows = [dataclasses.asdict(c) for c in replay.calibration]
    _write_json(out_dir / "calibration.json", calibration_rows)
    _write_json(out_dir / "frontier.json", replay.frontier)
    (out_dir / "calibration_table.md").write_text(
        report_tables.calibration_table(calibration_rows) + "\n"
    )
    (out_dir / "frontier_table.md").write_text(report_tables.frontier_table(replay.frontier) + "\n")
    inside = sum(1 for c in replay.calibration if c.inside_ci)
    print(f"[replay] {inside}/{len(replay.calibration)} policies realized within the OPE DR CI")

    # 4. Estimator-breakage study (known-truth tabular simulator).
    breakage = run_breakage_study(ope)
    _write_json(out_dir / "breakage.json", breakage)
    (out_dir / "breakage_table.md").write_text(report_tables.breakage_table(breakage) + "\n")
    print("[breakage] swept logging overlap; wrote bias/variance table")

    # 5. Figures (optional; skipped cleanly if the analysis extra is absent).
    try:
        from specialist_router.analysis import plots

        plots.plot_frontier(replay.frontier, out_dir / "frontier.png")
        plots.plot_calibration(calibration_rows, out_dir / "calibration.png")
        print("[figures] wrote frontier.png and calibration.png")
    except ImportError:
        print("[figures] matplotlib not installed (analysis extra) — skipping plots")

    print(f"Done. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
