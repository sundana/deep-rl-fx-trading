"""SB3 features extractor that runs observations through PatchTST.

The forex env emits a flat observation: window*n_features + 3 (position metadata).
This extractor splits the two parts, runs the windowed features through the
PatchTST encoder, and concatenates the per-step metadata back on for the
policy/value heads.
"""
from __future__ import annotations

import warnings

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .model import PatchTSTConfig, PatchTSTEncoder
from .pretrained import load_encoder_state


class PatchTSTFeaturesExtractor(BaseFeaturesExtractor):
    META_DIM = 3  # position, unrealized pnl, bars_in_trade — appended by ForexEnv

    def __init__(
        self,
        observation_space: spaces.Box,
        patchtst_cfg_dict: dict,
        pretrained_path: str | None = None,
        freeze: bool = False,
    ):
        cfg = PatchTSTConfig(**patchtst_cfg_dict)
        super().__init__(observation_space, features_dim=cfg.embed_dim + self.META_DIM)
        self.cfg = cfg
        self._window_floats = cfg.window * cfg.n_features
        expected = self._window_floats + self.META_DIM
        if observation_space.shape != (expected,):
            raise ValueError(
                f"Obs shape {observation_space.shape} != expected ({expected},). "
                f"Check window={cfg.window} and n_features={cfg.n_features}."
            )
        self.encoder = PatchTSTEncoder(cfg)
        self.freeze = bool(freeze)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in self.encoder.parameters())
            print(f"[patchtst-extractor] encoder FROZEN ({n:,} params)")

    def _load_pretrained(self, path: str) -> None:
        try:
            state, saved_cfg = load_encoder_state(path)
        except Exception as e:
            warnings.warn(f"[patchtst-extractor] failed to read {path}: {e}. Using random init.")
            return
        if (saved_cfg.window, saved_cfg.n_features, saved_cfg.embed_dim) != (
            self.cfg.window, self.cfg.n_features, self.cfg.embed_dim
        ):
            warnings.warn(
                f"[patchtst-extractor] config mismatch loading {path}: "
                f"saved=(window={saved_cfg.window}, n_features={saved_cfg.n_features}, "
                f"embed_dim={saved_cfg.embed_dim}) vs current=(window={self.cfg.window}, "
                f"n_features={self.cfg.n_features}, embed_dim={self.cfg.embed_dim}). Using random init."
            )
            return
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"[patchtst-extractor] partial load from {path}: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )
        else:
            print(f"[patchtst-extractor] loaded pretrained encoder from {path}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        window_flat = obs[:, : self._window_floats]
        meta = obs[:, self._window_floats :]
        x = window_flat.reshape(B, self.cfg.window, self.cfg.n_features)
        if self.freeze:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        return torch.cat([z, meta], dim=1)
