"""Integration tests for the stub-backed routing service and the metered episode runner."""

from __future__ import annotations

from pathlib import Path

from specialist_router.config import Config, RouterConfig, ServingConfig
from specialist_router.env.database import Dataset, build_dataset
from specialist_router.env.reference_agents import OracleAgent
from specialist_router.env.tasks import generate_tasks
from specialist_router.env.tools import ToolContext
from specialist_router.router.features import EntityVocab, Featurizer
from specialist_router.router.logger import LoggedDataset
from specialist_router.router.policies import UniformPolicy
from specialist_router.serving.service import (
    EpisodeArmRunner,
    RouterService,
    SimulatedArmRunner,
    build_stub_runners,
    collect_decisions,
)


def _service(env: Config, router: RouterConfig, serving: ServingConfig) -> RouterService:
    feat = Featurizer(router.features, EntityVocab.from_db_config(env.db))
    runners = build_stub_runners(serving.seed, serving.stub.local, serving.stub.api, env.verifier)
    return RouterService(feat, UniformPolicy(), runners, router.reward, router.seed)


def test_uniform_traffic_is_well_formed(
    env_config: Config, router_config: RouterConfig, serving_config: ServingConfig
) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    service = _service(env_config, router_config, serving_config)
    decisions = collect_decisions(service, tasks, seed=router_config.seed)

    assert len(decisions) == len(tasks)
    assert all(d.propensity == 0.5 for d in decisions)  # Uniform logging policy
    assert all(d.quality in (0, 1) for d in decisions)
    assert all(d.feature_dim == decisions[0].feature_dim for d in decisions)

    data = LoggedDataset.from_decisions(decisions)
    assert bool(data.arm_mask("local").any()) and bool(data.arm_mask("api").any())


def test_stub_api_arm_is_stronger_than_local(
    env_config: Config, serving_config: ServingConfig
) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    tasks = generate_tasks(dataset, env_config)
    local = SimulatedArmRunner("local", serving_config.stub.local, env_config.verifier, 0)
    api = SimulatedArmRunner("api", serving_config.stub.api, env_config.verifier, 0)
    local_quality = sum(local.run(t).quality for t in tasks)
    api_quality = sum(api.run(t).quality for t in tasks)
    assert api_quality > local_quality  # the frontier arm should win overall


def test_stub_outcomes_are_reproducible(env_config: Config, serving_config: ServingConfig) -> None:
    dataset = build_dataset(env_config.db, env_config.seed)
    task = generate_tasks(dataset, env_config)[0]
    runner = SimulatedArmRunner("api", serving_config.stub.api, env_config.verifier, 7)
    first, second = runner.run(task), runner.run(task)
    assert (first.quality, first.cost_usd, first.latency_s) == (
        second.quality,
        second.cost_usd,
        second.latency_s,
    )


def test_episode_arm_runner_meters_a_real_episode(
    env_config: Config, mini_dataset: Dataset, mini_db_file: Path
) -> None:
    # Task and DB must come from the same dataset so the oracle's reference SQL matches ground
    # truth — the frozen mini fixture (mini_dataset backs mini_db_file).
    task = generate_tasks(mini_dataset, env_config)[0]
    tools = ToolContext(mini_db_file, env_config.tools.run_sql, env_config.tools.python_calc)
    runner = EpisodeArmRunner(
        arm="local",
        version="oracle-test",
        agent_factory=OracleAgent,
        tools=tools,
        episode_config=env_config.episode,
        verifier_config=env_config.verifier,
        cost_fn=lambda _agent: 0.001,
    )
    outcome = runner.run(task)
    assert outcome.quality == 1  # the oracle solves it
    assert outcome.latency_s >= 0.0
    assert outcome.stop_reason == "final_answer"
    tools.close()
