"""Unit tests for the training guardrails: top-K checkpoints, format-drift, spot safety."""

from __future__ import annotations

import signal

from specialist_router.training.callbacks import (
    FormatRateTrend,
    SpotInterruptGuard,
    TopKCheckpointRegistry,
)


def test_topk_keeps_best_and_evicts_rest() -> None:
    reg = TopKCheckpointRegistry(k=2)
    assert reg.consider(100, 0.5, "a") == []
    assert reg.consider(200, 0.7, "b") == []
    evicted = reg.consider(300, 0.6, "c")  # full; 0.6 < 0.7 and 0.6 > 0.5 -> evict "a"
    assert evicted == ["a"]
    kept = {r.path for r in reg.best}
    assert kept == {"b", "c"}
    assert reg.best[0].path == "b"  # best score first


def test_topk_tie_keeps_incumbent() -> None:
    reg = TopKCheckpointRegistry(k=1)
    reg.consider(1, 0.8, "first")
    evicted = reg.consider(2, 0.8, "second")  # equal score does not displace incumbent
    assert evicted == ["second"]
    assert reg.best[0].path == "first"


def test_format_drift_warns_when_success_up_and_format_down() -> None:
    trend = FormatRateTrend(window=2, min_delta=0.02)
    assert trend.record(0, success=0.40, format_rate=0.95) is None
    assert trend.record(50, success=0.45, format_rate=0.95) is None
    # success rose (0.40 -> 0.55) while format-rate fell (0.95 -> 0.80): the drift signal (rider 2).
    warning = trend.record(100, success=0.55, format_rate=0.80)
    assert warning is not None
    assert "format drift" in warning


def test_format_trend_no_warning_when_both_rise() -> None:
    trend = FormatRateTrend(window=1, min_delta=0.02)
    trend.record(0, success=0.40, format_rate=0.80)
    assert trend.record(50, success=0.60, format_rate=0.90) is None  # format also rose -> healthy


def test_spot_guard_flushes_on_signal() -> None:
    flushed = {"n": 0}
    before = signal.getsignal(signal.SIGTERM)

    guard = SpotInterruptGuard(lambda: flushed.__setitem__("n", flushed["n"] + 1))
    guard.install()
    try:
        # Simulate a preemption SIGTERM by invoking the installed handler directly.
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
    finally:
        guard.restore()

    assert guard.triggered is True
    assert flushed["n"] == 1  # a checkpoint was flushed before exit
    assert signal.getsignal(signal.SIGTERM) is before  # handlers restored
