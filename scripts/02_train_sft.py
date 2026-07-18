#!/usr/bin/env python
"""Supervised training of the generative retriever."""

from __future__ import annotations

import argparse

from recgrpo.train.sft import train_sft
from recgrpo.utils import get_logger, load_config, set_seed
from recgrpo.workspace import load_workspace

logger = get_logger("sft")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train["seed"])

    ws = load_workspace(cfg.processed_dir)
    val_examples = ws.eval_examples(
        "val",
        cfg.data["max_history_items"],
        subsample=cfg.eval["val_subsample"],
        seed=cfg.train["seed"],
    )
    logger.info("catalog %d items, %d validation targets per epoch", ws.n_items, len(val_examples))

    train_sft(
        train_df=ws.split.train,
        val_examples=val_examples,
        item_tokens=ws.item_tokens,
        trie=ws.trie,
        cold_mask=ws.cold_mask,
        cfg=cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()
