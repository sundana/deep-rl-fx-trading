"""Supervised pretraining for the PatchTST encoder.

Trains a PatchTSTForecaster (encoder + linear head) to predict the next-N
close-price log-returns from a 64-bar feature window. Saves the encoder
state_dict + config to disk; PPO then loads that checkpoint via
PatchTSTFeaturesExtractor.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_loader import load_market_data, split_data, standardize_with
from src.patchtst import PatchTSTConfig, PatchTSTForecaster, save_encoder
from src.patchtst.dataset import ForecastDataset
from src.sessions import filter_data_by_sessions


_ENCODER_KEYS = (
    "patch_len", "stride", "d_model", "n_heads", "depth",
    "dropout", "head_dim", "embed_dim",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Path for the best-by-val-loss encoder checkpoint. "
                             "Default: models/patchtst/PatchTST_<n>/encoder_best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--session", type=str, default=None,
                        help="Filter to specific sessions: 'asia', 'london', 'newyork', or comma-separated.")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pcfg = cfg.get("patchtst", {}) or {}
    seed = cfg["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    epochs = args.epochs or pcfg.get("pretrain_epochs", 30)
    batch_size = args.batch_size or pcfg.get("pretrain_batch_size", 256)
    lr = args.lr if args.lr is not None else pcfg.get("pretrain_lr", 3e-4)
    weight_decay = pcfg.get("pretrain_weight_decay", 1e-4)
    horizon = args.horizon or pcfg.get("horizon", 8)
    device = args.device or cfg["train"].get("device", "cuda")
    window = cfg["env"]["window_size"]

    print("[data] loading market data…")
    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, _ = split_data(data, cfg["data"]["train_split"], cfg["data"]["val_split"])

    if args.session:
        sessions = [s.strip().lower() for s in args.session.split(",")]
        print(f"[data] filtering to sessions: {', '.join(sessions)}")
        train_d = filter_data_by_sessions(train_d, sessions)
        val_d = filter_data_by_sessions(val_d, sessions)

    train_d, val_d = standardize_with(train_d, val_d)
    n_features = int(train_d.features.shape[1])
    print(f"[data] train={len(train_d)} bars  val={len(val_d)} bars  "
          f"features={n_features}  window={window}  horizon={horizon}")

    # Build PatchTST config from yaml overrides; runtime values for window/n_features
    enc_overrides = {k: pcfg[k] for k in _ENCODER_KEYS if k in pcfg}
    ptst_cfg = PatchTSTConfig(window=window, n_features=n_features, **enc_overrides)

    train_ds = ForecastDataset(train_d, window=window, horizon=horizon)
    val_ds = ForecastDataset(val_d, window=window, horizon=horizon)
    pin = device != "cpu"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=pin,
    )
    print(f"[data] train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # Auto-numbered run name (parity with SB3 RecurrentPPO_N convention)
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("PatchTST_")]
    run_n = len(existing) + 1
    run_name = f"PatchTST_{run_n}"
    print(f"[pretrain] run name: {run_name}")

    model_dir = Path(cfg["paths"]["model_dir"]) / "patchtst" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir / run_name))

    model = PatchTSTForecaster(ptst_cfg, horizon=horizon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"[model] total={n_params:,}  encoder={enc_params:,}  cfg={ptst_cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_path = Path(args.output) if args.output else (model_dir / "encoder_best.pt")
    last_path = model_dir / "encoder_last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_sum, n_train = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_sum += loss.item() * X.shape[0]
            n_train += X.shape[0]
        train_loss = train_sum / max(1, n_train)

        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(X)
                val_sum += loss_fn(pred, y).item() * X.shape[0]
                n_val += X.shape[0]
        val_loss = val_sum / max(1, n_val)

        dt = time.time() - t0
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        flag = " * best" if improved else ""
        print(f"[epoch {epoch:3d}/{epochs}] train_loss={train_loss:.6e}  "
              f"val_loss={val_loss:.6e}  ({dt:.1f}s){flag}")

        save_encoder(model.encoder, ptst_cfg, last_path)
        if improved:
            save_encoder(model.encoder, ptst_cfg, out_path)

    print(f"[pretrain] done. best val_loss={best_val:.6e}")
    print(f"[pretrain] best encoder -> {out_path}")
    print(f"[pretrain] last encoder -> {last_path}")
    writer.close()


if __name__ == "__main__":
    main()
