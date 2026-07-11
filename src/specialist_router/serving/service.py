"""The routing service core: featurize → choose an arm → run it → verify → reward → log.

This is the backend-agnostic heart of ``/solve``, deliberately free of any web framework so it is
unit/integration-testable on CPU with no server. It selects an arm by sampling the *policy's own
action distribution* and logs that action's propensity, so the logged propensity is exactly
consistent with how the action was drawn — the invariant importance-sampling relies on.

Two arm runners implement the ``ArmRunner`` protocol:

* :class:`SimulatedArmRunner` — the CPU/CI stub: quality is a seeded Bernoulli (scored through the
  *real* verifier), cost/latency are seeded draws. No GPU, no API, no database needed. Every
  artifact it produces is stub/CPU-simulator data (real-model numbers arrive in Phase 4).
* :class:`EpisodeArmRunner` — the real path: runs a live agent through the sandboxed episode loop
  and measures wall-clock latency and token cost.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from specialist_router.agents.base import Agent
from specialist_router.config import RewardConfig, StubArmConfig, VerifierConfig
from specialist_router.env.episode import run_and_verify
from specialist_router.env.records import (
    ARMS,
    AnswerType,
    AnswerValue,
    Arm,
    Difficulty,
    RouterDecision,
    Task,
    Verdict,
)
from specialist_router.env.tools import ToolContext
from specialist_router.env.verifier import verify
from specialist_router.router.features import Featurizer
from specialist_router.router.policies import Policy
from specialist_router.router.reward import compute_reward

_DIFFICULTY_LEVEL: dict[Difficulty, int] = {"easy": 0, "med": 1, "hard": 2}


def synthetic_timestamp(index: int) -> str:
    """A deterministic ISO-8601 timestamp (one virtual second per decision).

    Real wall-clock time would make committed artifacts non-reproducible; a synthetic clock keeps
    the decision logs byte-stable across runs so ``repro-phase2`` reproduces them exactly.
    """
    minutes, seconds = divmod(index, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"2026-07-11T{hours:02d}:{minutes:02d}:{seconds:02d}+{days:03d}d"


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """What running one arm on one task yielded (the reward inputs)."""

    quality: int
    cost_usd: float
    latency_s: float
    verdict: Verdict | None = None
    stop_reason: str | None = None


class ArmRunner(Protocol):
    """Runs a single arm on a task and returns its :class:`ArmOutcome`."""

    arm: Arm
    version: str

    def run(self, task: Task) -> ArmOutcome:
        """Execute the arm on ``task``."""
        ...


def _wrong_answer(answer_type: AnswerType) -> AnswerValue:
    """A shape-valid but wrong answer for a given type (so it flows through the real verifier)."""
    if answer_type is AnswerType.LIST_STR:
        return ["__wrong__"]
    if answer_type is AnswerType.INTEGER:
        return -1
    return -999999.0


def _decision_rng(seed: int, task_id: str, arm: Arm) -> np.random.Generator:
    """A per-(task, arm) RNG so a simulated outcome is reproducible and order-independent."""
    digest = hashlib.blake2b(f"{seed}:{task_id}:{arm}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


class SimulatedArmRunner:
    """A seeded quality/cost/latency simulator for one arm (CPU dev; no GPU/API/database).

    Quality is drawn from a Bernoulli whose success probability decays with the *true* difficulty
    (the simulator may see it; the router may not) and then scored through the real verifier by
    submitting either the ground-truth answer or a wrong one — so ``quality`` is genuinely a
    verifier verdict, not a shortcut.
    """

    def __init__(
        self, arm: Arm, config: StubArmConfig, verifier_config: VerifierConfig, seed: int
    ) -> None:
        """Bind the simulated arm to its generative profile and the verifier tolerances."""
        self.arm = arm
        self.version = f"stub-{arm}"
        self._config = config
        self._verifier_config = verifier_config
        self._seed = seed

    def run(self, task: Task) -> ArmOutcome:
        """Simulate solving ``task`` with this arm and return the outcome."""
        rng = _decision_rng(self._seed, task.task_id, self.arm)
        level = _DIFFICULTY_LEVEL[task.difficulty]
        raw_p = self._config.base_quality - self._config.difficulty_penalty * level
        p_correct = min(max(raw_p, 0.0), 1.0)
        correct = bool(rng.random() < p_correct)
        submitted = task.expected if correct else _wrong_answer(task.answer_type)
        verdict = verify(task.expected, submitted, task.answer_type, self._verifier_config)
        cost = max(0.0, float(rng.normal(self._config.cost_usd_mean, self._config.cost_usd_std)))
        latency = max(
            1e-3, float(rng.normal(self._config.latency_s_mean, self._config.latency_s_std))
        )
        return ArmOutcome(
            quality=int(verdict.correct),
            cost_usd=cost,
            latency_s=latency,
            verdict=verdict,
            stop_reason="final_answer",
        )


class EpisodeArmRunner:
    """Runs a live agent through the sandboxed episode loop, measuring latency and token cost."""

    def __init__(
        self,
        arm: Arm,
        version: str,
        agent_factory: Callable[[], Agent],
        tools: ToolContext,
        episode_config: object,
        verifier_config: VerifierConfig,
        cost_fn: Callable[[Agent], float],
    ) -> None:
        """Bind the real arm to its agent factory, sandbox, and cost model."""
        self.arm = arm
        self.version = version
        self._agent_factory = agent_factory
        self._tools = tools
        self._episode_config = episode_config
        self._verifier_config = verifier_config
        self._cost_fn = cost_fn

    def run(self, task: Task) -> ArmOutcome:
        """Run one episode on ``task`` and return the measured outcome."""
        from specialist_router.config import EpisodeConfig

        assert isinstance(self._episode_config, EpisodeConfig)
        agent = self._agent_factory()
        start = time.perf_counter()
        trajectory, verdict = run_and_verify(
            task, agent, self._tools, self._episode_config, self._verifier_config
        )
        latency = time.perf_counter() - start
        return ArmOutcome(
            quality=int(verdict.correct),
            cost_usd=max(0.0, self._cost_fn(agent)),
            latency_s=latency,
            verdict=verdict,
            stop_reason=trajectory.stop_reason,
        )


class RouterService:
    """Bind a featurizer, a policy, and one runner per arm into a per-task routing decision."""

    def __init__(
        self,
        featurizer: Featurizer,
        policy: Policy,
        runners: dict[Arm, ArmRunner],
        reward_config: RewardConfig,
        seed: int,
    ) -> None:
        """Construct the service; ``runners`` must cover every arm in :data:`ARMS`."""
        missing = [arm for arm in ARMS if arm not in runners]
        if missing:
            raise ValueError(f"runners missing for arms: {missing}")
        self._featurizer = featurizer
        self._policy = policy
        self._runners = runners
        self._reward_config = reward_config
        self._seed = seed
        self._agent_versions = {arm: runners[arm].version for arm in ARMS}

    def decide(
        self, task: Task, rng: np.random.Generator, decision_index: int, timestamp: str
    ) -> RouterDecision:
        """Route one task: sample an arm from the policy, run it, and build the logged decision."""
        features = self._featurizer.transform(task.question)
        propensities = self._policy.propensities(features)
        probs = np.array([propensities[a] for a in ARMS], dtype=np.float64)
        arm = ARMS[int(rng.choice(len(ARMS), p=probs / probs.sum()))]

        outcome = self._runners[arm].run(task)
        breakdown = compute_reward(
            outcome.quality, outcome.cost_usd, outcome.latency_s, self._reward_config
        )
        return RouterDecision(
            decision_id=f"dec-{decision_index:07d}",
            task_id=task.task_id,
            template_id=task.template_id,
            difficulty=task.difficulty,
            feature_names=self._featurizer.feature_names,
            feature_vector=[float(v) for v in features],
            feature_dim=self._featurizer.feature_dim,
            policy_name=self._policy.name,
            policy_version=self._policy.version,
            policy_params_hash=self._policy.params_hash,
            action=arm,
            propensity=float(propensities[arm]),
            all_propensities={a: float(propensities[a]) for a in ARMS},
            quality=breakdown.quality,
            cost_usd=breakdown.cost_usd,
            latency_s=breakdown.latency_s,
            cost_norm=breakdown.cost_norm,
            latency_norm=breakdown.latency_norm,
            reward=breakdown.reward,
            reward_lambda=self._reward_config.lambda_cost,
            reward_mu=self._reward_config.mu_latency,
            cost_ref_usd=self._reward_config.cost_ref_usd,
            latency_ref_s=self._reward_config.latency_ref_s,
            agent_versions={str(a): v for a, v in self._agent_versions.items()},
            seed=self._seed,
            timestamp=timestamp,
        )


def collect_decisions(
    service: RouterService, tasks: list[Task], seed: int, start_index: int = 0
) -> list[RouterDecision]:
    """Route every task through ``service`` under one seeded RNG stream, in order.

    This is the shared inner loop for both traffic generation (logging policy) and replay-A/B
    (a deployed candidate policy). Timestamps come from :func:`synthetic_timestamp` so runs are
    reproducible.
    """
    rng = np.random.default_rng(seed)
    return [
        service.decide(task, rng, start_index + i, synthetic_timestamp(start_index + i))
        for i, task in enumerate(tasks)
    ]


def build_stub_runners(
    serving_seed: int,
    stub_local: StubArmConfig,
    stub_api: StubArmConfig,
    verifier_config: VerifierConfig,
) -> dict[Arm, ArmRunner]:
    """Construct the two simulated arm runners used for CPU traffic generation and repro."""
    return {
        "local": SimulatedArmRunner("local", stub_local, verifier_config, serving_seed),
        "api": SimulatedArmRunner("api", stub_api, verifier_config, serving_seed),
    }
