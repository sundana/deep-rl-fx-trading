"""Save/load helpers for PatchTST encoder weights.

The pretraining script saves only the encoder backbone (not the forecast head)
along with its config dataclass, so the SB3 features extractor can rebuild the
exact same architecture and verify shape compatibility before loading weights.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .model import PatchTSTConfig, PatchTSTEncoder


def save_encoder(encoder: PatchTSTEncoder, cfg: PatchTSTConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "config": asdict(cfg)}, path)


def load_encoder_state(path: str | Path) -> tuple[dict, PatchTSTConfig]:
    """Returns (state_dict, config) — caller is responsible for loading into a module."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["state_dict"], PatchTSTConfig(**blob["config"])
