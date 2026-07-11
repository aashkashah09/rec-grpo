"""Matplotlib figures for the Phase-2 artifacts (optional ``analysis`` extra; lazy import).

Two figures, both generated only from committed JSON result rows (``CLAUDE.md``: never hand-made):
the cost/quality frontier and the replay calibration (OPE-predicted vs realized). Importing this
module is cheap; matplotlib is imported inside the functions so CI (which does not install the
``analysis`` extra) never needs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_STUB_NOTE = "stub / CPU-simulator (Phase 2)"


def plot_frontier(rows: Sequence[dict[str, object]], out_path: str | Path) -> None:
    """Scatter realized mean cost vs mean quality, one point per deployed policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        x = float(row["mean_cost_usd"])  # type: ignore[arg-type]
        y = float(row["mean_quality"])  # type: ignore[arg-type]
        ax.scatter(x, y)
        ax.annotate(
            str(row["policy"]), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    ax.set_xlabel("mean cost per decision (USD)")
    ax.set_ylabel("mean quality (verifier pass rate)")
    ax.set_title(f"Cost / quality frontier — {_STUB_NOTE}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(rows: Sequence[dict[str, object]], out_path: str | Path) -> None:
    """Plot OPE-predicted value (with CI error bars) against realized value; y=x is perfect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    predicted = [float(r["ope_point"]) for r in rows]  # type: ignore[arg-type]
    realized = [float(r["realized_value"]) for r in rows]  # type: ignore[arg-type]
    lo = [float(r["ope_point"]) - float(r["ope_lo"]) for r in rows]  # type: ignore[arg-type]
    hi = [float(r["ope_hi"]) - float(r["ope_point"]) for r in rows]  # type: ignore[arg-type]
    ax.errorbar(predicted, realized, xerr=[lo, hi], fmt="o", capsize=3)
    for row, px, py in zip(rows, predicted, realized, strict=True):
        ax.annotate(
            str(row["policy_name"]), (px, py), fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    lim = [min(predicted + realized) - 0.05, max(predicted + realized) + 0.05]
    ax.plot(lim, lim, "--", color="gray", alpha=0.6)
    ax.set_xlabel("OPE-predicted value (DR)")
    ax.set_ylabel("realized value (replay)")
    ax.set_title(f"OPE calibration — {_STUB_NOTE}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
