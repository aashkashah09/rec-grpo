"""RQ-VAE training loop."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..utils.logging import JsonlLogger, get_logger
from .rqvae import RQVAE

logger = get_logger(__name__)


def _lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_rqvae(
    embeddings: np.ndarray,
    cfg,
    log_path: str | Path | None = None,
    device: str | None = None,
) -> RQVAE:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)

    model = RQVAE(
        input_dim=cfg.rqvae["input_dim"],
        hidden_dims=cfg.rqvae["hidden_dims"],
        latent_dim=cfg.rqvae["latent_dim"],
        n_levels=cfg.rqvae["n_levels"],
        codebook_size=cfg.rqvae["codebook_size"],
        commitment_weight=cfg.rqvae["commitment_weight"],
        dropout=cfg.rqvae.get("dropout", 0.0),
    ).to(device)

    if cfg.rqvae.get("kmeans_init", True):
        with torch.no_grad():
            sample = x[torch.randperm(len(x), device=device)[:8192]]
            model.quantizer.init_from_data(model.encoder(sample), iters=cfg.rqvae["kmeans_iters"])
        logger.info("initialised codebooks from %d encoded items", min(len(x), 8192))

    steps = cfg.train["steps"]
    batch_size = cfg.train["batch_size"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"])
    sink = JsonlLogger(log_path) if log_path else None
    steps_per_epoch = max(1, len(x) // batch_size)

    model.train()
    for step in range(steps):
        lr = _lr_at(step, steps, cfg.train["lr"], cfg.train["warmup_steps"])
        for group in opt.param_groups:
            group["lr"] = lr

        idx = torch.randint(0, len(x), (batch_size,), device=device)
        out = model(x[idx])

        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train["grad_clip"])
        opt.step()

        if (step + 1) % steps_per_epoch == 0:
            with torch.no_grad():
                z = model.encoder(x)
                residual = z
                for level in model.quantizer.levels:
                    n_dead = level.reseed_dead_codes(residual, cfg.rqvae["dead_code_threshold"])
                    q, _, _ = level(residual)
                    residual = residual - q
                    if n_dead:
                        logger.debug("step %d: reseeded %d dead codes", step + 1, n_dead)

        if sink and (step % cfg.train["log_every"] == 0 or step == steps - 1):
            sink.log(
                step=step,
                lr=round(lr, 8),
                recon_loss=round(float(out["recon_loss"]), 6),
                quant_loss=round(float(out["quant_loss"]), 6),
                loss=round(float(out["loss"]), 6),
                active_codes=model.quantizer.active_codes,
            )

    if sink:
        sink.close()

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dict(cfg.rqvae)}, ckpt_dir / "rqvae.pt")
    logger.info("saved RQ-VAE to %s", ckpt_dir / "rqvae.pt")
    return model
