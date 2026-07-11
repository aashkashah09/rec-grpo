"""Propensity-logged decision I/O and the array view estimators/policies consume.

Two responsibilities, both trust-critical (a corrupted or zero-propensity log silently breaks
every downstream estimator, so nothing here fails silently):

* :class:`DecisionLogger` writes validated :class:`RouterDecision` records as versioned JSONL and
  reads them back, asserting the logging invariant ``propensity > 0`` on every line.
* :class:`LoggedDataset` is the dense numpy view of a decision list — the single structure both
  the bandit policies (for offline fitting) and the OPE estimators operate on, so feature/action/
  propensity/reward alignment is defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import numpy.typing as npt

from specialist_router.env.records import ARMS, Arm, RouterDecision


class LogIntegrityError(Exception):
    """Raised when a logged decision violates an invariant estimators rely on."""


_ARM_INDEX: dict[Arm, int] = {arm: i for i, arm in enumerate(ARMS)}


@dataclass(frozen=True, slots=True)
class LoggedDataset:
    """A dense, aligned numpy view over a list of :class:`RouterDecision` records."""

    features: npt.NDArray[np.float64]
    """``(n, d)`` context matrix."""

    action_index: npt.NDArray[np.intp]
    """``(n,)`` index of the logged action into :data:`ARMS`."""

    propensity: npt.NDArray[np.float64]
    """``(n,)`` logging propensity ``π₀(action | x)`` — strictly positive."""

    reward: npt.NDArray[np.float64]
    """``(n,)`` composite reward."""

    quality: npt.NDArray[np.intp]
    """``(n,)`` binary verifier verdict."""

    cost_norm: npt.NDArray[np.float64]
    latency_norm: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        """Number of logged decisions."""
        return int(self.features.shape[0])

    @property
    def d(self) -> int:
        """Feature dimensionality."""
        return int(self.features.shape[1])

    def arm_mask(self, arm: Arm) -> npt.NDArray[np.bool_]:
        """Boolean mask selecting rows whose logged action is ``arm``."""
        return cast("npt.NDArray[np.bool_]", self.action_index == _ARM_INDEX[arm])

    @classmethod
    def from_decisions(cls, decisions: list[RouterDecision]) -> LoggedDataset:
        """Build the aligned arrays from decision records.

        Raises:
            LogIntegrityError: If the list is empty, feature widths disagree, or any propensity is
                non-positive (which would make an importance weight infinite).
        """
        if not decisions:
            raise LogIntegrityError("cannot build a LoggedDataset from zero decisions")
        dim = decisions[0].feature_dim
        features = np.empty((len(decisions), dim), dtype=np.float64)
        action_index = np.empty(len(decisions), dtype=np.intp)
        propensity = np.empty(len(decisions), dtype=np.float64)
        reward = np.empty(len(decisions), dtype=np.float64)
        quality = np.empty(len(decisions), dtype=np.intp)
        cost_norm = np.empty(len(decisions), dtype=np.float64)
        latency_norm = np.empty(len(decisions), dtype=np.float64)
        for i, decision in enumerate(decisions):
            if decision.feature_dim != dim or len(decision.feature_vector) != dim:
                raise LogIntegrityError(
                    f"decision {decision.decision_id} has feature width "
                    f"{len(decision.feature_vector)} (dim={decision.feature_dim}); expected {dim}"
                )
            if not decision.propensity > 0.0:
                raise LogIntegrityError(
                    f"decision {decision.decision_id} has non-positive propensity "
                    f"{decision.propensity}"
                )
            features[i] = decision.feature_vector
            action_index[i] = _ARM_INDEX[decision.action]
            propensity[i] = decision.propensity
            reward[i] = decision.reward
            quality[i] = decision.quality
            cost_norm[i] = decision.cost_norm
            latency_norm[i] = decision.latency_norm
        return cls(
            features=features,
            action_index=action_index,
            propensity=propensity,
            reward=reward,
            quality=quality,
            cost_norm=cost_norm,
            latency_norm=latency_norm,
        )


class DecisionLogger:
    """Append :class:`RouterDecision` records to a JSONL file as a context manager."""

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for writing (parent directories are created)."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def __enter__(self) -> DecisionLogger:
        """Open the underlying file handle."""
        self._handle = self._path.open("w")
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the underlying file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, decision: RouterDecision) -> None:
        """Validate and append one decision as a JSON line.

        Raises:
            LogIntegrityError: If the decision's propensity is non-positive.
            RuntimeError: If the logger is used outside its context manager.
        """
        if self._handle is None:
            raise RuntimeError("DecisionLogger must be used as a context manager")
        if not decision.propensity > 0.0:
            raise LogIntegrityError(
                f"refusing to log decision {decision.decision_id} with non-positive propensity "
                f"{decision.propensity}"
            )
        self._handle.write(decision.model_dump_json() + "\n")


def read_decisions(path: str | Path) -> list[RouterDecision]:
    """Read and validate a decisions JSONL file into :class:`RouterDecision` records.

    Raises:
        LogIntegrityError: If any line carries a non-positive propensity.
    """
    decisions: list[RouterDecision] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        decision = RouterDecision.model_validate_json(line)
        if not decision.propensity > 0.0:
            raise LogIntegrityError(
                f"decision {decision.decision_id} in {path} has non-positive propensity"
            )
        decisions.append(decision)
    return decisions
