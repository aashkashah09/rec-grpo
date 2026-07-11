"""A tiny tabular contextual-bandit simulator with an analytically known policy value.

This is the ground-truth oracle the property tests check the estimators against: a finite set of
discrete contexts with known reward means and a known logging policy. Because the true value of
any target policy is available in closed form, we can assert the statistical properties the
estimators are supposed to have (IPS unbiased in expectation, SNIPS/DR lower variance, DR robust
to a misspecified outcome model).

It is intentionally not tied to the SQL environment — it exercises the estimator maths in
isolation, at a scale where the truth is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from specialist_router.env.records import ARMS, Arm
from specialist_router.router.logger import LoggedDataset

_N_ARMS = len(ARMS)


class TabularPolicy:
    """A policy defined by an explicit ``(n_contexts, n_arms)`` probability table.

    Contexts are one-hot encoded, so the acting context is recovered as the argmax of the feature
    vector. Implements the :class:`~specialist_router.router.policies.Policy` protocol so the OPE
    estimators can consume it unchanged.
    """

    def __init__(self, table: npt.NDArray[np.float64], name: str = "tabular") -> None:
        """Wrap a row-stochastic probability table (rows sum to 1)."""
        self.name = name
        self.version = "1"
        self._table = table

    @property
    def params_hash(self) -> str:
        """A constant tag — the table is fixed at construction."""
        return "tabular"

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Look up the action distribution for the one-hot context ``x``."""
        context = int(np.argmax(x))
        return {arm: float(self._table[context, i]) for i, arm in enumerate(ARMS)}

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Sample an action for context ``x`` and return it with its propensity."""
        propensities = self.propensities(x)
        probs = np.array([propensities[a] for a in ARMS], dtype=np.float64)
        idx = int(rng.choice(_N_ARMS, p=probs / probs.sum()))
        return ARMS[idx], propensities[ARMS[idx]]


@dataclass(frozen=True, slots=True)
class BanditSimulator:
    """A finite contextual bandit with Bernoulli rewards and a known logging policy."""

    context_probs: npt.NDArray[np.float64]
    """``(K,)`` distribution over contexts."""

    reward_means: npt.NDArray[np.float64]
    """``(K, n_arms)`` true ``E[reward | context, arm]`` in ``[0, 1]``."""

    logging: npt.NDArray[np.float64]
    """``(K, n_arms)`` logging policy ``π₀(a | context)`` — strictly positive for overlap."""

    @property
    def n_contexts(self) -> int:
        """Number of discrete contexts."""
        return int(self.context_probs.shape[0])

    def true_value(self, target_table: npt.NDArray[np.float64]) -> float:
        """The exact value ``E_x[ Σ_a π_e(a|x) · reward_mean(x, a) ]`` of a target policy table."""
        per_context = np.sum(target_table * self.reward_means, axis=1)
        return float(np.sum(self.context_probs * per_context))

    def sample(self, n: int, seed: int) -> LoggedDataset:
        """Draw ``n`` logged decisions under the logging policy as a :class:`LoggedDataset`.

        Rewards are Bernoulli; ``quality`` mirrors the reward and ``cost_norm``/``latency_norm``
        are zero (the estimator maths under test does not depend on the cost/latency channel).
        """
        rng = np.random.default_rng(seed)
        contexts = rng.choice(self.n_contexts, size=n, p=self.context_probs)
        features = np.eye(self.n_contexts, dtype=np.float64)[contexts]
        action_index = np.empty(n, dtype=np.intp)
        propensity = np.empty(n, dtype=np.float64)
        reward = np.empty(n, dtype=np.float64)
        for i, context in enumerate(contexts):
            probs = self.logging[context]
            a = int(rng.choice(_N_ARMS, p=probs))
            action_index[i] = a
            propensity[i] = probs[a]
            reward[i] = float(rng.random() < self.reward_means[context, a])
        return LoggedDataset(
            features=features,
            action_index=action_index,
            propensity=propensity,
            reward=reward,
            quality=reward.astype(np.intp),
            cost_norm=np.zeros(n, dtype=np.float64),
            latency_norm=np.zeros(n, dtype=np.float64),
        )
