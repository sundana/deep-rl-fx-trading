"""Sliding-window forecasting dataset for PatchTST supervised pretraining.

Yields (X, y):
    X: (window, n_features) standardized features ending at bar t-1
    y: (horizon,) cumulative log-return of close from t-1 to t-1+h, h in 1..horizon

This mirrors the env's observation construction (which slices the same
features array up to the current bar) so the encoder learns on the same
input distribution it will see during PPO rollouts.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data_loader import MarketData


class ForecastDataset(Dataset):
    def __init__(self, data: MarketData, window: int, horizon: int):
        if len(data) < window + horizon + 1:
            raise ValueError(
                f"Need at least window+horizon+1={window + horizon + 1} bars; got {len(data)}."
            )
        self.window = int(window)
        self.horizon = int(horizon)
        self.features = data.features  # (T, F) standardized float32
        self.close = data.close.astype(np.float32)
        self._n = len(data) - self.window - self.horizon

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.features[i : i + self.window]
        anchor = float(self.close[i + self.window - 1])
        if anchor <= 0:
            anchor = 1e-12
        future = self.close[i + self.window : i + self.window + self.horizon]
        y = np.log(future / anchor).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)
