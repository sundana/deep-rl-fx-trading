"""PatchTST encoder + forecasting head.

Channel-independent variant: each input channel (feature) is patched and
processed by a shared transformer; the per-channel summaries are concatenated
and projected to a fixed embed_dim.

The encoder is the transferable backbone — used both for supervised forecast
pretraining (wrapped by `PatchTSTForecaster`) and as the SB3 features extractor
backbone during PPO training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PatchTSTConfig:
    window: int = 64
    n_features: int = 19
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    depth: int = 3
    dropout: float = 0.1
    head_dim: int = 16
    embed_dim: int = 128

    @property
    def n_patches(self) -> int:
        return (self.window - self.patch_len) // self.stride + 1


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST encoder.

    Input:  (B, L, F) where L = window, F = n_features
    Output: (B, embed_dim)
    """

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        if (cfg.window - cfg.patch_len) % cfg.stride != 0:
            # not strictly required, but keeps n_patches integer & avoids dropped tail
            pass
        self.cfg = cfg
        n_patches = cfg.n_patches
        if n_patches < 1:
            raise ValueError(
                f"n_patches={n_patches} <= 0; check window={cfg.window}, "
                f"patch_len={cfg.patch_len}, stride={cfg.stride}."
            )

        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, cfg.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.flatten_head = nn.Linear(n_patches * cfg.d_model, cfg.head_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.n_features * cfg.head_dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if L != self.cfg.window or F != self.cfg.n_features:
            raise ValueError(
                f"Expected (B, {self.cfg.window}, {self.cfg.n_features}); got {tuple(x.shape)}."
            )
        # (B, L, F) -> (B, F, L) -> (B*F, L)
        x = x.permute(0, 2, 1).reshape(B * F, L)
        # patch via unfold: (B*F, n_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        z = self.patch_embed(patches) + self.pos_embed         # (B*F, n_patches, d_model)
        z = self.transformer(z)                                 # (B*F, n_patches, d_model)
        z = z.flatten(1)                                        # (B*F, n_patches*d_model)
        z = self.flatten_head(z)                                # (B*F, head_dim)
        z = z.reshape(B, F * self.cfg.head_dim)                 # (B, F*head_dim)
        z = self.dropout(z)
        return self.proj(z)                                     # (B, embed_dim)


class PatchTSTForecaster(nn.Module):
    """Encoder + linear forecast head for supervised pretraining.

    Predicts `horizon` future values from a single (window, n_features) input.
    The head is discarded when transferring the encoder into the SB3 extractor.
    """

    def __init__(self, cfg: PatchTSTConfig, horizon: int):
        super().__init__()
        self.encoder = PatchTSTEncoder(cfg)
        self.horizon = horizon
        self.head = nn.Linear(cfg.embed_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
