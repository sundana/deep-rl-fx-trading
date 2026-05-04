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


def save_forecaster(forecaster, cfg: PatchTSTConfig, horizon: int, path: str | Path) -> None:
    """Save the full encoder+head so the forecast head can be evaluated later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": forecaster.state_dict(),
        "config": asdict(cfg),
        "horizon": horizon,
    }, path)


def load_encoder_state(path: str | Path) -> tuple[dict, PatchTSTConfig]:
    """Returns (state_dict, config) — caller is responsible for loading into a module."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["state_dict"], PatchTSTConfig(**blob["config"])


def load_forecaster(path: str | Path, device: str = "cpu"):
    """Load a full PatchTSTForecaster (encoder + head) from a forecaster checkpoint."""
    from .model import PatchTSTForecaster
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = PatchTSTConfig(**blob["config"])
    horizon = blob["horizon"]
    model = PatchTSTForecaster(cfg, horizon=horizon)
    model.load_state_dict(blob["state_dict"])
    return model, cfg, horizon
