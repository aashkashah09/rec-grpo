"""Supervised stage: next-item prediction over the training window."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SequenceDataset, collate_sequences
from ..eval.evaluate import evaluate
from ..model.transformer import SemanticIDTransformer, count_parameters
from ..model.trie import CatalogTrie
from ..semid.vocab import PAD
from ..utils.logging import JsonlLogger, get_logger

logger = get_logger(__name__)


def _lr_at(step: int, total: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_sft(
    train_df,
    val_examples,
    item_tokens: np.ndarray,
    trie: CatalogTrie,
    cold_mask: np.ndarray,
    cfg,
    device: str | None = None,
) -> SemanticIDTransformer:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = SemanticIDTransformer(
        d_model=cfg.model["d_model"],
        n_layers=cfg.model["n_layers"],
        n_heads=cfg.model["n_heads"],
        d_ff=cfg.model["d_ff"],
        max_seq_len=cfg.model["max_seq_len"],
        dropout=cfg.model["dropout"],
        tie_embeddings=cfg.model["tie_embeddings"],
    ).to(device)
    logger.info("model: %s parameters", f"{count_parameters(model):,}")

    dataset = SequenceDataset(
        train_df,
        item_tokens,
        max_history_items=cfg.data["max_history_items"],
        min_history_items=cfg.data["min_history_items"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train["batch_size"],
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=cfg.train.get("num_workers", 4),
        drop_last=True,
        pin_memory=device == "cuda",
    )
    logger.info("%d training sequences, %d steps/epoch", len(dataset), len(loader))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train["lr"],
        betas=tuple(cfg.train["betas"]),
        weight_decay=cfg.train["weight_decay"],
    )
    total_steps = len(loader) * cfg.train["epochs"]

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log = JsonlLogger(Path(cfg.results_dir) / "logs" / "sft_train.jsonl")
    val_log = JsonlLogger(Path(cfg.results_dir) / "logs" / "sft_val.jsonl")

    step = 0
    best_ndcg = -1.0
    epochs_since_best = 0
    start = time.time()

    for epoch in range(1, cfg.train["epochs"] + 1):
        model.train()
        for batch in loader:
            lr = _lr_at(step, total_steps, cfg.train["lr"], cfg.train["min_lr"], cfg.train["warmup_steps"])
            for group in opt.param_groups:
                group["lr"] = lr

            tokens = batch["tokens"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)

            logits = model(tokens[:, :-1], attn_mask=attn_mask[:, :-1])
            targets = tokens[:, 1:].masked_fill(~loss_mask[:, 1:], PAD)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=PAD,
                label_smoothing=cfg.train["label_smoothing"],
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train["grad_clip"])
            opt.step()

            if step % cfg.train["log_every"] == 0:
                value = float(loss.detach())
                train_log.log(
                    step=step,
                    epoch=epoch,
                    loss=round(value, 5),
                    ppl=round(math.exp(min(value, 20)), 4),
                    lr=round(lr, 8),
                    grad_norm=round(float(grad_norm), 4),
                    elapsed=round(time.time() - start, 1),
                )
            step += 1

        if epoch % cfg.train["eval_every_epochs"] == 0:
            metrics, _ = evaluate(
                model,
                val_examples,
                item_tokens,
                trie,
                cold_mask,
                beam_size=cfg.eval["beam_size"],
                ks=tuple(cfg.eval["topk"]),
                device=device,
                desc=f"val e{epoch}",
            )
            ndcg10 = metrics["all"]["ndcg@10"]
            val_log.log(
                epoch=epoch,
                step=step,
                **{
                    "recall@10": round(metrics["all"]["recall@10"], 6),
                    "ndcg@10": round(ndcg10, 6),
                    "cold_recall@10": round(metrics.get("cold", {}).get("recall@10", 0.0), 6),
                },
            )
            logger.info(
                "epoch %d | val recall@10 %.4f ndcg@10 %.4f cold %.4f",
                epoch,
                metrics["all"]["recall@10"],
                ndcg10,
                metrics.get("cold", {}).get("recall@10", 0.0),
            )

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                epochs_since_best = 0
                torch.save(
                    {"model": model.state_dict(), "config": dict(cfg.model), "epoch": epoch},
                    ckpt_dir / "best.pt",
                )
            else:
                epochs_since_best += 1
                if epochs_since_best >= cfg.train["early_stop_patience"]:
                    logger.info("no val improvement for %d epochs, stopping", epochs_since_best)
                    break

    train_log.close()
    val_log.close()

    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    logger.info("restored best checkpoint from epoch %d (val ndcg@10 %.4f)", state["epoch"], best_ndcg)
    return model
