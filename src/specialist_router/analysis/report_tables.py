"""Pure Markdown-table formatters for the Phase-2 artifacts (no plotting, no I/O).

Every figure/table in the docs is generated from committed artifacts (``CLAUDE.md``) — these
functions turn the JSON result rows produced by :mod:`specialist_router.analysis.pipeline` into
Markdown tables for the README/report. They are pure and unit-tested; the matplotlib figures live
in :mod:`specialist_router.analysis.plots`.
"""

from __future__ import annotations

from collections.abc import Sequence

_STUB_CAPTION = (
    "_Stub-agent / CPU-simulator results — not real-model numbers (those land in Phase 4)._"
)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table from headers and pre-formatted string cells."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join([head, sep, body]) if rows else "\n".join([head, sep])


def _f(value: object, digits: int = 3) -> str:
    """Format a numeric cell to fixed precision (passing through non-numbers)."""
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)


def ope_table(rows: Sequence[dict[str, object]]) -> str:
    """Format the OPE results table (one row per policy) with a stub-data caption."""
    headers = ["policy", "IPS", "SNIPS", "DM", "DR", "DR 95% CI", "ESS frac", "max w"]
    body = [
        [
            row["policy"],
            _f(row["ips"]),
            _f(row["snips"]),
            _f(row["direct_method"]),
            _f(row["doubly_robust"]),
            f"[{_f(row['dr_lo'])}, {_f(row['dr_hi'])}]",
            _f(row["ess_fraction"], 2),
            _f(row["max_weight"], 2),
        ]
        for row in rows
    ]
    return markdown_table(headers, body) + "\n\n" + _STUB_CAPTION


def calibration_table(rows: Sequence[dict[str, object]]) -> str:
    """Format the replay-A/B calibration table (OPE prediction vs realized value)."""
    headers = ["policy", "OPE DR", "DR 95% CI", "realized", "n", "in CI?", "abs err"]
    body = [
        [
            row["policy_name"],
            _f(row["ope_point"]),
            f"[{_f(row['ope_lo'])}, {_f(row['ope_hi'])}]",
            _f(row["realized_value"]),
            row["realized_n"],
            "yes" if row["inside_ci"] else "no",
            _f(row["abs_error"]),
        ]
        for row in rows
    ]
    return markdown_table(headers, body) + "\n\n" + _STUB_CAPTION


def frontier_table(rows: Sequence[dict[str, object]]) -> str:
    """Format the cost/quality frontier table (realized, per deployed policy)."""
    headers = ["policy", "quality", "cost $/dec", "latency s", "reward", "% to api"]
    body = [
        [
            row["policy"],
            _f(row["mean_quality"]),
            _f(row["mean_cost_usd"], 5),
            _f(row["mean_latency_s"], 2),
            _f(row["mean_reward"]),
            _f(row["frac_api"], 2),
        ]
        for row in rows
    ]
    return markdown_table(headers, body) + "\n\n" + _STUB_CAPTION


def lambda_mu_sweep_table(rows: Sequence[dict[str, object]]) -> str:
    """Format the λ/μ sweep: DR of the best learned router vs always_api per grid cell."""
    headers = [
        "λ",
        "μ",
        "always_api DR",
        "best learned DR",
        "winner",
        "margin",
        "learned wins?",
    ]
    body = [
        [
            _f(row["lambda"], 2),
            _f(row["mu"], 2),
            _f(row["dr_always_api"]),
            _f(row["best_learned_dr"]),
            row["best_learned_policy"],
            _f(row["margin_vs_api"]),
            "yes" if row["learned_beats_api"] else "no",
        ]
        for row in rows
    ]
    return markdown_table(headers, body) + "\n\n" + _STUB_CAPTION


def breakage_table(rows: Sequence[dict[str, object]]) -> str:
    """Format the estimator-breakage study table (bias/variance vs shrinking overlap)."""
    headers = [
        "min π₀",
        "true V",
        "IPS bias",
        "IPS std",
        "SNIPS bias",
        "SNIPS std",
        "DR bias",
        "DR std",
        "ESS frac",
    ]
    body = [
        [
            _f(row["logging_min_propensity"], 2),
            _f(row["true_value"]),
            _f(row["ips_bias"]),
            _f(row["ips_std"]),
            _f(row["snips_bias"]),
            _f(row["snips_std"]),
            _f(row["dr_bias"]),
            _f(row["dr_std"]),
            _f(row["mean_ess_fraction"], 2),
        ]
        for row in rows
    ]
    return markdown_table(headers, body)
