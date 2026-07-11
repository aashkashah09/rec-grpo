"""Contextual-bandit routing policies over the two arms (``local`` vs ``api``).

Every policy exposes the same small surface — :meth:`propensities` (the full action distribution
``π(a | x)``, which OPE needs as the numerator/target) and :meth:`select` (sample an action and
return it with its propensity, which traffic generation needs). Keeping the distribution explicit
and full-support (no policy ever assigns probability 0 to an arm it might, in expectation, take)
is what makes the logs valid for importance-sampling estimators.

* :class:`UniformPolicy` — the logging policy: 0.5 / 0.5, no training. Guarantees overlap.
* :class:`EpsilonGreedyPolicy` — ridge value model per arm; greedy with ε uniform exploration.
* :class:`LinUCBPolicy` — disjoint per-arm ridge with a UCB bonus, softmaxed into a logged
  propensity so the deployed policy is stochastic (and thus a valid target/proposal).
* :class:`ThompsonLogisticPolicy` — Bayesian logistic model of the *binary quality* per arm; the
  arm value subtracts the expected cost/latency penalty (the same λ/μ as the reward), and action
  probabilities are estimated by Monte-Carlo posterior sampling.

All fitting is offline from a :class:`~specialist_router.router.logger.LoggedDataset`; all policies
are pure given their fitted parameters and the supplied RNG (reproducible).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from specialist_router.config import (
    EpsilonGreedyConfig,
    LinUCBConfig,
    RewardConfig,
    RouterConfig,
    ThompsonConfig,
)
from specialist_router.env.records import ARMS, Arm
from specialist_router.router.logger import LoggedDataset

_N_ARMS = len(ARMS)


def _hash_params(*arrays: npt.NDArray[np.float64]) -> str:
    """Stable short hash of fitted numeric parameters (for the decision log)."""
    digest = hashlib.blake2b(digest_size=8)
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.float64).round(6).tobytes())
    return digest.hexdigest()


def _sample(propensities: dict[Arm, float], rng: np.random.Generator) -> tuple[Arm, float]:
    """Sample an arm from a categorical action distribution and return ``(arm, its prob)``."""
    probs = np.array([propensities[a] for a in ARMS], dtype=np.float64)
    idx = int(rng.choice(_N_ARMS, p=probs / probs.sum()))
    arm = ARMS[idx]
    return arm, propensities[arm]


class Policy(Protocol):
    """A routing policy: a full, positive action distribution over the arms given a context."""

    name: str
    version: str

    @property
    def params_hash(self) -> str:
        """A stable hash of the policy's parameters, logged with every decision."""
        ...

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Return ``π(a | x)`` for every arm (values sum to 1, all strictly positive)."""
        ...

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Sample an action for context ``x`` and return it with its propensity."""
        ...


class UniformPolicy:
    """The logging policy: pick each arm with probability ``1 / n_arms``, independent of context."""

    name = "uniform"
    version = "1"

    def __init__(self) -> None:
        """Construct the (parameter-free) uniform logging policy."""
        self._p = 1.0 / _N_ARMS

    @property
    def params_hash(self) -> str:
        """Constant — the uniform policy has no learned parameters."""
        return "uniform"

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Return the uniform distribution over arms."""
        return {arm: self._p for arm in ARMS}

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Sample an arm uniformly at random."""
        return _sample(self.propensities(x), rng)


class ConstantPolicy:
    """A deterministic reference policy that always picks one fixed arm.

    Used only as an OPE *target* (a frontier-only or specialist-only baseline point) — never as a
    logging policy, since it has no support on the other arm. Rows where the logged action differs
    simply receive importance weight 0, which is exactly correct for a deterministic target.
    """

    def __init__(self, arm: Arm) -> None:
        """Create the always-``arm`` policy."""
        self.arm = arm
        self.name = f"always_{arm}"
        self.version = "1"

    @property
    def params_hash(self) -> str:
        """Constant — no learned parameters."""
        return f"const-{self.arm}"

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Put all mass on the fixed arm."""
        return {a: (1.0 if a == self.arm else 0.0) for a in ARMS}

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Always return the fixed arm with propensity 1."""
        return self.arm, 1.0


def _ridge_fit(
    features: npt.NDArray[np.float64], targets: npt.NDArray[np.float64], ridge_lambda: float
) -> npt.NDArray[np.float64]:
    """Closed-form ridge weights ``w = (XᵀX + λI)⁻¹ Xᵀy`` (λ > 0 keeps the system PD)."""
    dim = features.shape[1]
    gram = features.T @ features + ridge_lambda * np.eye(dim)
    return np.linalg.solve(gram, features.T @ targets).astype(np.float64, copy=False)


class EpsilonGreedyPolicy:
    """Ridge value model per arm; act greedily w.r.t. predicted reward with ε uniform exploration.

    Propensity is exact: the greedy arm gets ``1 − ε + ε/n_arms`` and every other arm ``ε/n_arms``.
    """

    name = "epsilon_greedy"
    version = "1"

    def __init__(self, config: EpsilonGreedyConfig) -> None:
        """Bind hyperparameters; call :meth:`fit` before use."""
        self._config = config
        self._weights: npt.NDArray[np.float64] | None = None  # (n_arms, d)

    def fit(self, data: LoggedDataset) -> EpsilonGreedyPolicy:
        """Fit one ridge reward model per arm from the logged decisions."""
        weights = np.zeros((_N_ARMS, data.d), dtype=np.float64)
        for i, arm in enumerate(ARMS):
            mask = data.arm_mask(arm)
            if bool(mask.any()):
                weights[i] = _ridge_fit(
                    data.features[mask], data.reward[mask], self._config.ridge_lambda
                )
        self._weights = weights
        return self

    def _predicted(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._weights is None:
            raise RuntimeError("EpsilonGreedyPolicy.fit must be called before use")
        return self._weights @ x

    @property
    def params_hash(self) -> str:
        """Hash of the fitted per-arm weights."""
        if self._weights is None:
            return "unfit"
        return _hash_params(self._weights)

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Exact ε-greedy action distribution for context ``x``."""
        greedy = int(np.argmax(self._predicted(x)))
        eps = self._config.epsilon
        floor = eps / _N_ARMS
        return {arm: floor + (1.0 - eps if i == greedy else 0.0) for i, arm in enumerate(ARMS)}

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Sample greedily-with-exploration and return the action and its propensity."""
        return _sample(self.propensities(x), rng)


class LinUCBPolicy:
    """Disjoint LinUCB: per-arm ridge posterior with a UCB bonus, softmaxed to a logged propensity.

    A deterministic argmax-UCB policy would assign some arms propensity 0, which is invalid for
    importance sampling. Softmaxing the UCB scores at a configured temperature yields a smooth,
    full-support action distribution while preserving LinUCB's optimism-driven preferences.
    """

    name = "linucb"
    version = "1"

    def __init__(self, config: LinUCBConfig) -> None:
        """Bind hyperparameters; call :meth:`fit` before use."""
        self._config = config
        self._a_inv: npt.NDArray[np.float64] | None = None  # (n_arms, d, d)
        self._theta: npt.NDArray[np.float64] | None = None  # (n_arms, d)

    def fit(self, data: LoggedDataset) -> LinUCBPolicy:
        """Accumulate per-arm ``A = λI + Σ xxᵀ`` and ``b = Σ r x``; store ``A⁻¹`` and ``θ``."""
        dim = data.d
        a_inv = np.zeros((_N_ARMS, dim, dim), dtype=np.float64)
        theta = np.zeros((_N_ARMS, dim), dtype=np.float64)
        for i, arm in enumerate(ARMS):
            mask = data.arm_mask(arm)
            amat = self._config.ridge_lambda * np.eye(dim)
            bvec = np.zeros(dim, dtype=np.float64)
            if bool(mask.any()):
                xa = data.features[mask]
                amat = amat + xa.T @ xa
                bvec = xa.T @ data.reward[mask]
            inv = np.linalg.inv(amat)
            a_inv[i] = inv
            theta[i] = inv @ bvec
        self._a_inv = a_inv
        self._theta = theta
        return self

    def _scores(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._a_inv is None or self._theta is None:
            raise RuntimeError("LinUCBPolicy.fit must be called before use")
        scores = np.empty(_N_ARMS, dtype=np.float64)
        for i in range(_N_ARMS):
            mean = float(self._theta[i] @ x)
            bonus = self._config.alpha * float(np.sqrt(max(x @ self._a_inv[i] @ x, 0.0)))
            scores[i] = mean + bonus
        return scores

    @property
    def params_hash(self) -> str:
        """Hash of the fitted per-arm ``θ`` vectors."""
        if self._theta is None:
            return "unfit"
        return _hash_params(self._theta)

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Softmax of the per-arm UCB scores at the configured temperature."""
        scores = self._scores(x) / self._config.temperature
        scores = scores - scores.max()
        weights = np.exp(scores)
        probs = weights / weights.sum()
        return {arm: float(probs[i]) for i, arm in enumerate(ARMS)}

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Sample from the softmaxed-UCB distribution and return the action and its propensity."""
        return _sample(self.propensities(x), rng)


def _sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _laplace_logistic(
    features: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    prior_variance: float,
    newton_steps: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit Bayesian logistic regression by Newton's method; return (MAP mean, Laplace covariance).

    A zero-mean Gaussian prior with the given variance regularises the fit (and defines the
    posterior when an arm's logged labels are all one class). Returns the MAP weights and the
    inverse Hessian at the mode (the Laplace-approximate posterior covariance).
    """
    dim = features.shape[1]
    prior_precision = np.eye(dim) / prior_variance
    w = np.zeros(dim, dtype=np.float64)
    hessian = prior_precision
    for _ in range(newton_steps):
        p = _sigmoid(features @ w)
        gradient = features.T @ (p - labels) + prior_precision @ w
        weights = p * (1.0 - p)
        hessian = features.T @ (features * weights[:, None]) + prior_precision
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    covariance = np.linalg.inv(hessian)
    return w.astype(np.float64, copy=False), covariance.astype(np.float64, copy=False)


class ThompsonLogisticPolicy:
    """Thompson sampling over per-arm Bayesian-logistic models of the binary quality reward.

    Each arm models ``P(quality = 1 | x)`` with a Laplace-approximated logistic posterior. An
    arm's *value* is its sampled success probability minus the expected cost/latency penalty
    (using the reward's own λ/μ and the arm's logged average normalized cost/latency), so the
    policy optimises the same composite objective as the reward. Action probabilities are
    Monte-Carlo estimates of ``P(arm = argmax value)`` under the joint posterior — a proper,
    full-support propensity for OPE.
    """

    name = "thompson_logistic"
    version = "1"

    def __init__(self, config: ThompsonConfig, reward_config: RewardConfig) -> None:
        """Bind hyperparameters and the reward weights; call :meth:`fit` before use."""
        self._config = config
        self._reward = reward_config
        self._mean: npt.NDArray[np.float64] | None = None  # (n_arms, d)
        self._chol: npt.NDArray[np.float64] | None = None  # (n_arms, d, d)
        self._penalty: npt.NDArray[np.float64] | None = None  # (n_arms,)

    def fit(self, data: LoggedDataset) -> ThompsonLogisticPolicy:
        """Fit a logistic posterior per arm and estimate each arm's expected penalty."""
        dim = data.d
        mean = np.zeros((_N_ARMS, dim), dtype=np.float64)
        chol = np.zeros((_N_ARMS, dim, dim), dtype=np.float64)
        penalty = np.zeros(_N_ARMS, dtype=np.float64)
        for i, arm in enumerate(ARMS):
            mask = data.arm_mask(arm)
            if bool(mask.any()):
                w, cov = _laplace_logistic(
                    data.features[mask],
                    data.quality[mask].astype(np.float64),
                    self._config.prior_variance,
                    self._config.newton_steps,
                )
                mean[i] = w
                chol[i] = _stable_cholesky(cov)
                penalty[i] = self._reward.lambda_cost * float(
                    data.cost_norm[mask].mean()
                ) + self._reward.mu_latency * float(data.latency_norm[mask].mean())
            else:
                chol[i] = np.sqrt(self._config.prior_variance) * np.eye(dim)
        self._mean = mean
        self._chol = chol
        self._penalty = penalty
        return self

    def _sampled_values(
        self, x: npt.NDArray[np.float64], rng: np.random.Generator, n_samples: int
    ) -> npt.NDArray[np.float64]:
        if self._mean is None or self._chol is None or self._penalty is None:
            raise RuntimeError("ThompsonLogisticPolicy.fit must be called before use")
        dim = x.shape[0]
        values = np.empty((n_samples, _N_ARMS), dtype=np.float64)
        for i in range(_N_ARMS):
            noise = rng.standard_normal((n_samples, dim)) @ self._chol[i].T
            logits = (self._mean[i] + noise) @ x
            values[:, i] = _sigmoid(logits) - self._penalty[i]
        return values

    def _propensities_with_rng(
        self, x: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> dict[Arm, float]:
        n = self._config.n_posterior_samples
        values = self._sampled_values(x, rng, n)
        winners = np.argmax(values, axis=1)
        counts = np.bincount(winners, minlength=_N_ARMS).astype(np.float64)
        # Laplace smoothing keeps every propensity strictly positive (valid for OPE).
        probs = (counts + 1.0) / (n + _N_ARMS)
        return {arm: float(probs[i]) for i, arm in enumerate(ARMS)}

    @property
    def params_hash(self) -> str:
        """Hash of the fitted posterior means and per-arm penalties."""
        if self._mean is None or self._penalty is None:
            return "unfit"
        return _hash_params(self._mean, self._penalty)

    def propensities(self, x: npt.NDArray[np.float64]) -> dict[Arm, float]:
        """Monte-Carlo action distribution using a fixed seed derived from ``x`` (deterministic).

        OPE needs ``π(a | x)`` to be a deterministic function of the context; the MC seed is
        therefore derived from the feature vector so repeated calls agree.
        """
        seed = int.from_bytes(
            hashlib.blake2b(
                np.ascontiguousarray(x, dtype=np.float64).round(6).tobytes(), digest_size=8
            ).digest(),
            "big",
        )
        return self._propensities_with_rng(x, np.random.default_rng(seed))

    def select(self, x: npt.NDArray[np.float64], rng: np.random.Generator) -> tuple[Arm, float]:
        """Take one Thompson draw (via the MC action distribution) and return it with propensity."""
        propensities = self._propensities_with_rng(x, rng)
        return _sample(propensities, rng)


def build_target_policies(config: RouterConfig, data: LoggedDataset) -> dict[str, Policy]:
    """Fit and return every candidate target policy the router evaluates, keyed by name.

    The learned policies (ε-greedy, LinUCB, Thompson) are fit offline on ``data``; the reference
    points (uniform, always-local, always-api) need no fitting. This single factory is shared by
    the OPE and replay scripts so both evaluate an identical policy set.
    """
    policies: list[Policy] = [
        UniformPolicy(),
        EpsilonGreedyPolicy(config.policies.epsilon_greedy).fit(data),
        LinUCBPolicy(config.policies.linucb).fit(data),
        ThompsonLogisticPolicy(config.policies.thompson, config.reward).fit(data),
        ConstantPolicy("local"),
        ConstantPolicy("api"),
    ]
    return {policy.name: policy for policy in policies}


def _stable_cholesky(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cholesky factor with escalating jitter, so posterior sampling never fails numerically."""
    dim = matrix.shape[0]
    matrix = 0.5 * (matrix + matrix.T)
    for exponent in range(-12, 0):
        try:
            chol = np.linalg.cholesky(matrix + (10.0**exponent) * np.eye(dim))
            return chol.astype(np.float64, copy=False)
        except np.linalg.LinAlgError:
            continue
    fallback = np.sqrt(np.diag(np.maximum(np.diag(matrix), 1e-9)))[:, None] * np.eye(dim)
    return cast("npt.NDArray[np.float64]", fallback)
