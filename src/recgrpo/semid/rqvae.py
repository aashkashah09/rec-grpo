"""Residual-quantized autoencoder over frozen text embeddings.

Three codebooks applied to successive residuals give each item a coarse-to-fine
code triple. Similar items share their first code, which is what lets the
decoder put probability mass on a region of the catalog before it has committed
to an item -- including on items it never saw during training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    """One codebook with straight-through gradients and a commitment penalty."""

    def __init__(self, codebook_size: int, dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.normal_(0, 0.02)
        self.register_buffer("usage", torch.zeros(codebook_size, dtype=torch.long))
        self._initialised = False

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, iters: int = 50) -> None:
        """k-means++ style init: sample distinct rows, then Lloyd iterations."""
        n = x.shape[0]
        if n < self.codebook_size:
            reps = self.codebook_size // n + 1
            x = x.repeat(reps, 1)[: self.codebook_size]
            centroids = x.clone()
        else:
            idx = torch.randperm(n, device=x.device)[: self.codebook_size]
            centroids = x[idx].clone()
        for _ in range(iters):
            assign = torch.cdist(x, centroids).argmin(dim=1)
            for k in range(self.codebook_size):
                members = x[assign == k]
                if len(members):
                    centroids[k] = members.mean(dim=0)
        self.embedding.weight.data.copy_(centroids)
        self._initialised = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices)

        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.training:
            self.usage.index_add_(0, indices, torch.ones_like(indices))

        quantized = x + (quantized - x).detach()  # straight-through
        return quantized, indices, loss

    @torch.no_grad()
    def reseed_dead_codes(self, x: torch.Tensor, threshold: int = 1) -> int:
        """Point unused codes at random data rows so they re-enter the competition."""
        dead = (self.usage < threshold).nonzero(as_tuple=True)[0]
        if len(dead):
            idx = torch.randint(0, x.shape[0], (len(dead),), device=x.device)
            self.embedding.weight.data[dead] = x[idx]
        self.usage.zero_()
        return int(len(dead))

    @property
    def active_codes(self) -> int:
        return int((self.usage > 0).sum())


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        n_levels: int = 3,
        codebook_size: int = 256,
        dim: int = 32,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.levels = nn.ModuleList(
            VectorQuantizer(codebook_size, dim, commitment_weight) for _ in range(n_levels)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        quantized = torch.zeros_like(x)
        losses = []
        codes = []
        for level in self.levels:
            q, idx, loss = level(residual)
            quantized = quantized + q
            residual = residual - q.detach()
            codes.append(idx)
            losses.append(loss)
        return quantized, torch.stack(codes, dim=1), torch.stack(losses).sum()

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, iters: int = 50) -> None:
        residual = x
        for level in self.levels:
            level.init_from_data(residual, iters=iters)
            q, _, _ = level(residual)
            residual = residual - q

    @property
    def active_codes(self) -> list[int]:
        return [level.active_codes for level in self.levels]


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        n_levels: int = 3,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout)
        self.decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim], dropout)
        self.quantizer = ResidualQuantizer(
            n_levels=n_levels,
            codebook_size=codebook_size,
            dim=latent_dim,
            commitment_weight=commitment_weight,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        z_q, codes, quant_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return {
            "recon": recon,
            "codes": codes,
            "recon_loss": recon_loss,
            "quant_loss": quant_loss,
            "loss": recon_loss + quant_loss,
        }

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        """(n, input_dim) -> (n, n_levels) integer codes."""
        self.eval()
        out = []
        for start in range(0, len(x), batch_size):
            chunk = x[start : start + batch_size]
            _, codes, _ = self.quantizer(self.encoder(chunk))
            out.append(codes.cpu().numpy())
        return np.concatenate(out, axis=0)
