"""Phase-2 orchestration: traffic → OPE → replay-A/B → estimator-breakage, all on CPU.

These are the reusable building blocks the ``scripts/`` CLIs and ``make repro-phase2`` call, kept
in one place so the pipeline is integration-testable end to end on the mini profile. Everything
here runs on the stub simulator — **every number it produces is stub/CPU-simulator data**;
real-model numbers arrive in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from specialist_router.config import Config, OpeConfig, RouterConfig, ServingConfig
from specialist_router.env.database import build_dataset
from specialist_router.env.records import ARMS, RouterDecision, Task
from specialist_router.env.tasks import generate_tasks
from specialist_router.ope.ci import bootstrap_intervals
from specialist_router.ope.estimators import evaluate_policy
from specialist_router.ope.replay import ReplayCalibration, calibrate
from specialist_router.ope.simulator import BanditSimulator, TabularPolicy
from specialist_router.router.features import EntityVocab, Featurizer
from specialist_router.router.logger import LoggedDataset
from specialist_router.router.policies import Policy, UniformPolicy, build_target_policies
from specialist_router.serving.service import (
    RouterService,
    build_stub_runners,
    collect_decisions,
)


def _tasks_for(env: Config, n_tasks: int, seed: int) -> tuple[list[Task], Config]:
    """Build ``n_tasks`` tasks under ``seed`` (returns the tasks and the config actually used)."""
    used = env.model_copy(
        update={"seed": seed, "tasks": env.tasks.model_copy(update={"n_tasks": n_tasks})}
    )
    dataset = build_dataset(used.db, used.seed)
    return generate_tasks(dataset, used), used


def _featurizer(env: Config, router: RouterConfig) -> Featurizer:
    return Featurizer(router.features, EntityVocab.from_db_config(env.db))


def generate_traffic(
    env: Config,
    router: RouterConfig,
    serving: ServingConfig,
    n_tasks: int,
    seed: int,
    stub_seed_offset: int = 0,
) -> list[RouterDecision]:
    """Route ``n_tasks`` tasks through the Uniform logging policy; return the logged decisions."""
    tasks, used = _tasks_for(env, n_tasks, seed)
    runners = build_stub_runners(
        serving.seed + stub_seed_offset, serving.stub.local, serving.stub.api, used.verifier
    )
    service = RouterService(
        _featurizer(env, router), UniformPolicy(), runners, router.reward, router.seed
    )
    return collect_decisions(service, tasks, seed=seed)


@dataclass(frozen=True, slots=True)
class OpeRun:
    """The result of evaluating every target policy off-policy on a logged dataset."""

    rows: list[dict[str, object]]
    policies: dict[str, Policy]
    data: LoggedDataset


def run_ope(decisions: list[RouterDecision], router: RouterConfig, ope: OpeConfig) -> OpeRun:
    """Fit and off-policy-evaluate every candidate policy; return summary rows + fitted policies."""
    data = LoggedDataset.from_decisions(decisions)
    policies = build_target_policies(router, data)
    rows: list[dict[str, object]] = []
    for name, policy in policies.items():
        evaluation = evaluate_policy(data, policy, ope)
        ci = bootstrap_intervals(evaluation, ope.n_bootstrap, ope.ci_alpha, ope.seed)
        rows.append(
            {
                "policy": name,
                "ips": evaluation.ips,
                "snips": evaluation.snips,
                "direct_method": evaluation.direct_method,
                "doubly_robust": evaluation.doubly_robust,
                "dr_lo": ci["doubly_robust"].lo,
                "dr_hi": ci["doubly_robust"].hi,
                "ess_fraction": evaluation.ess_fraction,
                "max_weight": evaluation.max_weight,
            }
        )
    return OpeRun(rows=rows, policies=policies, data=data)


def deploy_policy(
    env: Config,
    router: RouterConfig,
    serving: ServingConfig,
    policy: Policy,
    n_tasks: int,
    seed: int,
    start_index: int,
) -> list[RouterDecision]:
    """Deploy a fitted ``policy`` on a fresh task set and return the realized decisions."""
    tasks, used = _tasks_for(env, n_tasks, seed)
    runners = build_stub_runners(
        serving.seed + 10_000, serving.stub.local, serving.stub.api, used.verifier
    )
    service = RouterService(_featurizer(env, router), policy, runners, router.reward, router.seed)
    return collect_decisions(service, tasks, seed=seed, start_index=start_index)


@dataclass(frozen=True, slots=True)
class ReplayRun:
    """Replay-A/B calibration rows and the realized frontier points, per policy."""

    calibration: list[ReplayCalibration]
    frontier: list[dict[str, object]]


def run_replay(
    env: Config,
    router: RouterConfig,
    serving: ServingConfig,
    ope_run: OpeRun,
    logged_decisions: list[RouterDecision],
    ope: OpeConfig,
    n_tasks: int,
    seed: int,
) -> ReplayRun:
    """Deploy each candidate policy on fresh traffic; compare realized value to its OPE DR CI.

    The DR confidence interval used here is recomputed from the same logged dataset the policies
    were evaluated on, so the calibration is against the exact OPE prediction. Reward-scale parity
    between the logged and replayed decisions is enforced inside :func:`calibrate`.
    """
    calibrations: list[ReplayCalibration] = []
    frontier: list[dict[str, object]] = []
    for offset, (name, policy) in enumerate(ope_run.policies.items()):
        realized = deploy_policy(
            env,
            router,
            serving,
            policy,
            n_tasks,
            seed + offset,
            start_index=1_000_000 * (offset + 1),
        )
        evaluation = evaluate_policy(ope_run.data, policy, ope)
        dr_ci = bootstrap_intervals(evaluation, ope.n_bootstrap, ope.ci_alpha, ope.seed)[
            "doubly_robust"
        ]
        calibrations.append(calibrate(name, logged_decisions, realized, dr_ci, "doubly_robust"))
        frontier.append(
            {
                "policy": name,
                "mean_quality": float(np.mean([d.quality for d in realized])),
                "mean_cost_usd": float(np.mean([d.cost_usd for d in realized])),
                "mean_latency_s": float(np.mean([d.latency_s for d in realized])),
                "mean_reward": float(np.mean([d.reward for d in realized])),
                "frac_api": float(np.mean([d.action == "api" for d in realized])),
            }
        )
    return ReplayRun(calibration=calibrations, frontier=frontier)


def run_breakage_study(ope: OpeConfig, n: int = 4000, n_seeds: int = 20) -> list[dict[str, object]]:
    """Show how the estimators behave as logging overlap shrinks toward deterministic logging.

    Uses the tabular :class:`BanditSimulator` (known true value) so the bias and variance of each
    estimator are measured against ground truth. As the logging policy concentrates on one arm,
    overlap collapses: IPS variance explodes, SNIPS is steadier, and DR stays closest to the truth.
    """
    reward_means = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.55]], dtype=np.float64)
    context_probs = np.array([0.34, 0.33, 0.33], dtype=np.float64)
    target_table = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]], dtype=np.float64)
    target = TabularPolicy(target_table, name="target")

    rows: list[dict[str, object]] = []
    for epsilon in (0.5, 0.3, 0.15, 0.05, 0.01):
        logging = np.array([[epsilon, 1.0 - epsilon]] * 3, dtype=np.float64)
        sim = BanditSimulator(context_probs, reward_means, logging)
        truth = sim.true_value(target_table)
        ips_vals, snips_vals, dr_vals, ess_fracs = [], [], [], []
        for s in range(n_seeds):
            data = sim.sample(n, seed=ope.seed + s)
            evaluation = evaluate_policy(data, target, ope)
            ips_vals.append(evaluation.ips)
            snips_vals.append(evaluation.snips)
            dr_vals.append(evaluation.doubly_robust)
            ess_fracs.append(evaluation.ess_fraction)
        rows.append(
            {
                "logging_min_propensity": float(epsilon),
                "true_value": float(truth),
                "ips_bias": float(np.mean(ips_vals) - truth),
                "ips_std": float(np.std(ips_vals)),
                "snips_bias": float(np.mean(snips_vals) - truth),
                "snips_std": float(np.std(snips_vals)),
                "dr_bias": float(np.mean(dr_vals) - truth),
                "dr_std": float(np.std(dr_vals)),
                "mean_ess_fraction": float(np.mean(ess_fracs)),
            }
        )
    return rows


_LEARNED = ("epsilon_greedy", "linucb", "thompson_logistic")


def run_lambda_mu_sweep(
    decisions: list[RouterDecision],
    router: RouterConfig,
    ope: OpeConfig,
    lambdas: list[float],
    mus: list[float],
) -> list[dict[str, object]]:
    """Re-score the *existing* logs across a (λ, μ) grid and compare learned routers to always_api.

    The reward is a post-hoc function of the logged components, so no new traffic is needed: for
    each grid cell we recompute ``reward = quality − λ·cost_norm − μ·latency_norm`` from the logged
    ``cost_norm``/``latency_norm``, **refit** the learned policies under that reward (they optimise
    the objective they are scored against), and off-policy-evaluate every policy by DR. This
    surfaces the (λ, μ) regime where a learned router's DR value exceeds always_api's.
    """
    base = LoggedDataset.from_decisions(decisions)
    rows: list[dict[str, object]] = []
    for lam in lambdas:
        for mu in mus:
            reward = base.quality - lam * base.cost_norm - mu * base.latency_norm
            data = replace(base, reward=reward)
            cell_router = router.model_copy(
                update={
                    "reward": router.reward.model_copy(
                        update={"lambda_cost": lam, "mu_latency": mu}
                    )
                }
            )
            policies = build_target_policies(cell_router, data)
            dr = {
                name: evaluate_policy(data, policy, ope).doubly_robust
                for name, policy in policies.items()
            }
            best_name = max(_LEARNED, key=lambda n: dr[n])
            best_learned = dr[best_name]
            always_api = dr["always_api"]
            rows.append(
                {
                    "lambda": float(lam),
                    "mu": float(mu),
                    "dr_epsilon_greedy": dr["epsilon_greedy"],
                    "dr_linucb": dr["linucb"],
                    "dr_thompson_logistic": dr["thompson_logistic"],
                    "dr_always_api": always_api,
                    "dr_always_local": dr["always_local"],
                    "best_learned_policy": best_name,
                    "best_learned_dr": best_learned,
                    "margin_vs_api": best_learned - always_api,
                    "learned_beats_api": bool(best_learned > always_api),
                }
            )
    return rows


def frontier_arms_reference(decisions: list[RouterDecision]) -> dict[str, dict[str, float]]:
    """Per-arm realized quality/cost from the logged pool (frontier-only reference points)."""
    data_by_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = [d for d in decisions if d.action == arm]
        if rows:
            data_by_arm[arm] = {
                "mean_quality": float(np.mean([d.quality for d in rows])),
                "mean_cost_usd": float(np.mean([d.cost_usd for d in rows])),
            }
    return data_by_arm
