"""Training-loop guardrails: top-K checkpoints, format-rate drift, spot-safety, W&B logging.

The logic here is deliberately free of ``torch``/``transformers``/``wandb`` at import time (W&B is
imported lazily inside :class:`WandbRun`), so it is unit-testable on CPU/CI. The thin
``transformers.TrainerCallback`` that drives these on ``on_step_end`` / ``on_train_end`` is built
lazily in :mod:`specialist_router.training.grpo_run`, which is the only module that touches TRL.
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FrameType

from specialist_router.config import WandbConfig
from specialist_router.evaluation.harness import EvalReport


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """One retained checkpoint and the held-out score that earned its place."""

    step: int
    score: float
    path: str


class TopKCheckpointRegistry:
    """Keep the top-K checkpoints by held-out success, evicting the rest.

    This is the "best" set and is deliberately separate from the trainer's rolling "latest"
    checkpoints used for resume: the most recent step is not necessarily the best, and we want both
    guarantees (resume-ability *and* the K best models) without one clobbering the other.
    """

    def __init__(self, k: int) -> None:
        """Create a registry retaining the ``k`` highest-scoring checkpoints."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = k
        self._records: list[CheckpointRecord] = []

    def consider(self, step: int, score: float, path: str) -> list[str]:
        """Offer a checkpoint; return the paths of any checkpoints evicted as a result.

        Ties keep the incumbent (a strictly higher score is required to displace a full registry),
        so an unchanged score late in training does not churn the best set.
        """
        self._records.append(CheckpointRecord(step=step, score=score, path=path))
        self._records.sort(key=lambda r: (-r.score, r.step))
        keep = self._records[: self._k]
        evicted = [r.path for r in self._records[self._k :]]
        self._records = keep
        return evicted

    @property
    def best(self) -> list[CheckpointRecord]:
        """The retained checkpoints, best score first."""
        return list(self._records)


@dataclass(slots=True)
class FormatRateTrend:
    """Track (success, format-rate) over eval steps and flag format-gaming / protocol drift.

    Rider (Phase-3): a **falling format-rate while reward/success rises** is a warning sign — the
    model may be finding a degenerate path that still earns correctness reward while its tool-use
    episodes become less well-formed. :meth:`drift_warning` compares the latest eval against the one
    ``window`` evals earlier and returns a message when success rose by ``min_delta`` while
    format-rate fell by ``min_delta``.
    """

    window: int = 2
    min_delta: float = 0.02
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def record(self, step: int, success: float, format_rate: float) -> str | None:
        """Record one eval point and return a drift warning string if the pattern is present."""
        self.history.append((step, success, format_rate))
        return self.drift_warning()

    def drift_warning(self) -> str | None:
        """Return a warning if success rose while format-rate fell over the recent window."""
        if len(self.history) <= self.window:
            return None
        step_now, success_now, format_now = self.history[-1]
        _, success_then, format_then = self.history[-1 - self.window]
        rose = (success_now - success_then) > self.min_delta
        fell = (format_then - format_now) > self.min_delta
        if rose and fell:
            return (
                f"format drift at step {step_now}: success rose "
                f"{success_then:.3f}->{success_now:.3f} while format-rate fell "
                f"{format_then:.3f}->{format_now:.3f} (possible verifier gaming / protocol drift)"
            )
        return None


class SpotInterruptGuard:
    """Flush a checkpoint on SIGTERM/SIGINT so a spot/preemptible instance loses no progress.

    On a spot preemption the cloud sends SIGTERM (typically with a short grace period). We install a
    handler that records the interruption and invokes an injected ``flush`` (wired to save a
    checkpoint). The save runs synchronously inside the handler, so ``flush`` must be safe to call
    at an arbitrary point in the training loop.
    """

    def __init__(self, flush: Callable[[], None]) -> None:
        """Bind the guard to the checkpoint-flush callback."""
        self._flush = flush
        self.triggered = False
        self._previous: dict[int, object] = {}

    def install(self) -> None:
        """Install SIGTERM/SIGINT handlers (call once, from the main thread)."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Record the interruption and flush a checkpoint before exiting."""
        self.triggered = True
        self._flush()

    def restore(self) -> None:
        """Restore the previously installed signal handlers."""
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
        self._previous.clear()


class WandbRun:
    """A thin Weights & Biases wrapper (lazy import; a no-op when ``mode == 'disabled'``).

    Logs reward curves, per-template held-out success, and the format-rate trend. Kept minimal so
    the training entrypoint stays about wiring, not logging plumbing.
    """

    def __init__(self, config: WandbConfig, run_config: dict[str, object] | None = None) -> None:
        """Initialise the run (unless disabled); ``run_config`` is logged as the run's config."""
        self._enabled = config.mode != "disabled"
        self._run = None
        if not self._enabled:
            return
        import wandb  # lazy: optional 'training' extra

        self._run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            mode=config.mode,
            config=run_config or {},
        )

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a scalar metrics dict at an optional step."""
        if not self._enabled or self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_eval(self, report: EvalReport, step: int) -> None:
        """Log a held-out :class:`EvalReport`: overall + per-template success and format-rate."""
        if not self._enabled:
            return
        metrics: dict[str, float] = {
            "eval/overall_success": report.overall_success,
            "eval/format_rate": report.format_rate,
        }
        for template_id, stat in report.per_template.items():
            metrics[f"eval/success/{template_id}"] = stat.success
            metrics[f"eval/format_rate/{template_id}"] = stat.format_rate
        for difficulty, success in report.per_difficulty.items():
            metrics[f"eval/success_by_difficulty/{difficulty}"] = success
        self.log(metrics, step=step)

    def finish(self) -> None:
        """Close the run."""
        if self._enabled and self._run is not None:
            self._run.finish()
