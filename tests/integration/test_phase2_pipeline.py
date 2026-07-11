"""End-to-end integration of the Phase-2 pipeline on the mini profile (small, CPU-fast)."""

from __future__ import annotations

from specialist_router.analysis import report_tables
from specialist_router.analysis.pipeline import (
    generate_traffic,
    run_breakage_study,
    run_ope,
    run_replay,
)
from specialist_router.config import Config, OpeConfig, RouterConfig, ServingConfig


def test_traffic_to_ope_to_replay(
    env_config: Config,
    router_config: RouterConfig,
    ope_config: OpeConfig,
    serving_config: ServingConfig,
) -> None:
    logged = generate_traffic(env_config, router_config, serving_config, n_tasks=240, seed=1)
    assert len(logged) == 240

    ope_run = run_ope(logged, router_config, ope_config)
    assert {row["policy"] for row in ope_run.rows} == {
        "uniform",
        "epsilon_greedy",
        "linucb",
        "thompson_logistic",
        "always_local",
        "always_api",
    }
    # The Uniform target evaluated on Uniform logs has ESS == n (weights all 1).
    uniform_row = next(r for r in ope_run.rows if r["policy"] == "uniform")
    assert float(uniform_row["ess_fraction"]) > 0.99

    replay = run_replay(
        env_config, router_config, serving_config, ope_run, logged, ope_config, n_tasks=200, seed=2
    )
    assert len(replay.calibration) == 6
    assert len(replay.frontier) == 6
    # Reward-scale parity held (calibrate would have raised otherwise); tables render.
    rows = [
        {
            "policy_name": c.policy_name,
            "ope_point": c.ope_point,
            "ope_lo": c.ope_lo,
            "ope_hi": c.ope_hi,
            "realized_value": c.realized_value,
            "realized_n": c.realized_n,
            "inside_ci": c.inside_ci,
            "abs_error": c.abs_error,
        }
        for c in replay.calibration
    ]
    table = report_tables.calibration_table(rows)
    assert "realized" in table and "Phase 4" in table


def test_frontier_orders_arms_by_cost(
    env_config: Config,
    router_config: RouterConfig,
    ope_config: OpeConfig,
    serving_config: ServingConfig,
) -> None:
    logged = generate_traffic(env_config, router_config, serving_config, n_tasks=240, seed=3)
    ope_run = run_ope(logged, router_config, ope_config)
    replay = run_replay(
        env_config, router_config, serving_config, ope_run, logged, ope_config, n_tasks=200, seed=4
    )
    frontier = {row["policy"]: row for row in replay.frontier}
    # always_api is both higher quality and more expensive than always_local.
    assert frontier["always_api"]["mean_quality"] > frontier["always_local"]["mean_quality"]
    assert frontier["always_api"]["mean_cost_usd"] > frontier["always_local"]["mean_cost_usd"]


def test_breakage_study_shows_ips_variance_growth(ope_config: OpeConfig) -> None:
    rows = run_breakage_study(ope_config, n=1000, n_seeds=8)
    assert len(rows) == 5
    # As logging overlap shrinks (min propensity 0.5 -> 0.01): IPS variance grows, ESS collapses.
    assert float(rows[-1]["ips_std"]) > float(rows[0]["ips_std"])
    assert float(rows[-1]["mean_ess_fraction"]) < float(rows[0]["mean_ess_fraction"])
    # In the well-overlapped regime the estimate is essentially unbiased (propensities are exact).
    # (At the thinnest overlap, a few-seed mean is itself high-variance, so we only check the top.)
    assert abs(float(rows[0]["ips_bias"])) < 0.03
