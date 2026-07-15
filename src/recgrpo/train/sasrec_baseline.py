"""SASRec baseline, run through RecBole.

RecBole owns the model and the training loop; what matters here is that it is
fed the same interaction table and the same global temporal split, and scored
with full-catalog ranking, so its numbers sit next to the generative model's
without an asterisk. The cold slice is recomputed from RecBole's ranked lists
against the same cold-item mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def write_atomic_files(df: pd.DataFrame, out_dir: str | Path, dataset: str = "beauty") -> Path:
    """Emit RecBole's .inter format from the processed interaction table."""
    out_dir = Path(out_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}.inter"
    frame = pd.DataFrame(
        {
            "user_id:token": df["user"].to_numpy(),
            "item_id:token": df["item"].to_numpy(),
            "timestamp:float": df["timestamp"].to_numpy().astype(float),
        }
    )
    frame.to_csv(path, sep="\t", index=False)
    logger.info("wrote %d rows to %s", len(frame), path)
    return path


def run(config_file: str | Path) -> dict:
    """Train and evaluate SASRec, returning RecBole's test metrics."""
    from recbole.quick_start import run_recbole

    result = run_recbole(model="SASRec", dataset="beauty", config_file_list=[str(config_file)])
    return result


def cold_slice_from_ranked_lists(
    ranked_items: np.ndarray,
    targets: np.ndarray,
    cold_mask: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Recall@k restricted to targets with no training interaction.

    Takes RecBole's top-k item ids per test row so the baseline's cold number is
    computed exactly the way the generative model's is.
    """
    is_cold = cold_mask[targets]
    if not is_cold.any():
        return {"n": 0, "recall@10": 0.0, "hits@10": 0}
    hits = (ranked_items[is_cold, :k] == targets[is_cold, None]).any(axis=1)
    return {
        "n": int(is_cold.sum()),
        f"recall@{k}": float(hits.mean()),
        f"hits@{k}": int(hits.sum()),
    }
