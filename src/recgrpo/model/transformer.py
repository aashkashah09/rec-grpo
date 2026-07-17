"""Decoder-only transformer over semantic-ID tokens.

The user's history and the item being generated live in one token stream, so
recommendation is plain autoregressive decoding. Sized for a recommender, not
for a language model: the catalog is 12k items and the vocabulary is 1028
tokens, so most of the capacity goes into depth rather than embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..semid.vocab import PAD, VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.resid_drop(self.out_proj(out))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class SemanticIDTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        d_ff: int = 1280,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            # scaled init on residual projections, as in GPT-2
            if name.endswith("out_proj.weight") or name.endswith("ff.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence of {t} tokens exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos)[None])

        mask = None
        if attn_mask is not None:
            causal = torch.ones(t, t, dtype=torch.bool, device=tokens.device).tril()
            mask = causal[None, None] & attn_mask[:, None, None, :]
            # a fully padded row would softmax over nothing; let it attend to itself
            mask = mask | torch.eye(t, dtype=torch.bool, device=tokens.device)[None, None]

        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Tied weights are counted once."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total
