"""Evaluate a pretrained PatchTST encoder on held-out test data.

Metrics reported per horizon step h=1..8:
  - Directional accuracy  : sign(pred) == sign(actual), baseline = 50%
  - IC (Pearson r)        : correlation between pred and actual log-returns
  - RMSE                  : vs zero-prediction baseline
  - MAE
  - Mean predicted return : bias check — persistent positive bias explains long-only RL behavior
  - Pred std / Actual std : calibration — collapsed predictions mean no real conviction

Usage:
  python evaluate_patchtst.py                          # uses config.yaml defaults, test split
  python evaluate_patchtst.py --encoder models/patchtst/PatchTST_7/encoder_best.pt
  python evaluate_patchtst.py --split val
  python evaluate_patchtst.py --all-runs               # compare every PatchTST_N run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loader import load_market_data, split_data, standardize_with
from src.patchtst.dataset import ForecastDataset
from src.patchtst.pretrained import load_forecaster


def _resolve_forecaster_path(encoder_path: str | Path) -> Path:
    """Given an encoder_best.pt path, return the sibling forecaster_best.pt if it exists."""
    ep = Path(encoder_path)
    candidate = ep.with_name(ep.name.replace("encoder_", "forecaster_"))
    return candidate if candidate.exists() else ep


def _collect_predictions(
    encoder_path: str | Path,
    data,
    horizon: int,
    window: int,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (preds, actuals) each shaped (N, horizon).

    Prefers forecaster_*.pt (encoder+head) over encoder_*.pt (encoder only).
    Encoder-only checkpoints don't include the forecast head, so using them
    produces garbage predictions from a randomly-initialized head.
    """
    fpath = _resolve_forecaster_path(encoder_path)
    if fpath == Path(encoder_path):
        print(f"[warn] No forecaster checkpoint found alongside {encoder_path}.")
        print(f"[warn] Only encoder weights available — head is randomly initialized.")
        print(f"[warn] Re-run pretrain_patchtst.py to generate forecaster_best.pt.")
        raise FileNotFoundError(
            f"forecaster checkpoint not found. Expected: "
            f"{Path(encoder_path).with_name(Path(encoder_path).name.replace('encoder_', 'forecaster_'))}"
        )
    print(f"[eval] loading forecaster from {fpath}")
    model, cfg, ckpt_horizon = load_forecaster(fpath, device=device)
    if ckpt_horizon != horizon:
        print(f"[warn] config horizon={horizon} but checkpoint horizon={ckpt_horizon}; "
              f"using checkpoint value.")
        horizon = ckpt_horizon
    model.eval()

    ds = ForecastDataset(data, window=window, horizon=horizon)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds_list, actuals_list = [], []
    with torch.no_grad():
        for X, y in loader:
            preds_list.append(model(X.to(device)).cpu().numpy())
            actuals_list.append(y.numpy())

    return np.concatenate(preds_list), np.concatenate(actuals_list)


def _compute_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute per-horizon metrics. Both arrays: (N, horizon)."""
    horizon = preds.shape[1]
    results = {}

    for h in range(horizon):
        p = preds[:, h]
        a = actuals[:, h]

        rmse = float(np.sqrt(np.mean((p - a) ** 2)))
        mae = float(np.mean(np.abs(p - a)))
        rmse_baseline = float(np.sqrt(np.mean(a ** 2)))  # predict-zero baseline

        dir_acc = float(np.mean(np.sign(p) == np.sign(a)))

        # IC: Pearson correlation
        if p.std() > 1e-10 and a.std() > 1e-10:
            ic = float(np.corrcoef(p, a)[0, 1])
        else:
            ic = 0.0

        # Spearman rank IC (manual, avoids scipy dependency)
        def _rankdata(x):
            idx = np.argsort(x)
            ranks = np.empty_like(idx, dtype=float)
            ranks[idx] = np.arange(1, len(x) + 1)
            return ranks
        rp, ra = _rankdata(p), _rankdata(a)
        denom = rp.std() * ra.std()
        rank_ic = float(np.corrcoef(rp, ra)[0, 1]) if denom > 1e-10 else 0.0

        results[h + 1] = {
            "dir_acc":       dir_acc,
            "ic":            ic,
            "rank_ic":       rank_ic,
            "rmse":          rmse,
            "rmse_baseline": rmse_baseline,
            "rmse_skill":    1.0 - rmse / rmse_baseline if rmse_baseline > 0 else 0.0,
            "mae":           mae,
            "mean_pred":     float(p.mean()),
            "mean_actual":   float(a.mean()),
            "std_pred":      float(p.std()),
            "std_actual":    float(a.std()),
        }

    # Direction bias check: how often does model predict positive vs negative?
    step1_preds = preds[:, 0]
    results["_bias"] = {
        "pct_positive_pred": float(np.mean(step1_preds > 0)),
        "pct_negative_pred": float(np.mean(step1_preds < 0)),
        "mean_pred_overall": float(preds.mean()),
        "mean_actual_overall": float(actuals.mean()),
    }

    return results


def _print_report(metrics: dict, encoder_path: str, split: str, n_samples: int) -> None:
    print(f"\n{'='*65}")
    print(f"PatchTST Encoder Evaluation")
    print(f"  encoder : {encoder_path}")
    print(f"  split   : {split}  ({n_samples:,} windows)")
    print(f"{'='*65}")

    print(f"\n{'h':>3}  {'DirAcc':>7}  {'IC':>7}  {'RankIC':>7}  "
          f"{'RMSE':>9}  {'Baseline':>9}  {'Skill':>6}  {'MeanPred':>9}  {'PredStd':>8}")
    print("-" * 83)

    horizon = max(k for k in metrics if isinstance(k, int))
    for h in range(1, horizon + 1):
        m = metrics[h]
        skill_str = f"{m['rmse_skill']:+.3f}"
        print(
            f"{h:>3}  {m['dir_acc']:>7.3f}  {m['ic']:>+7.4f}  {m['rank_ic']:>+7.4f}  "
            f"{m['rmse']:>9.6f}  {m['rmse_baseline']:>9.6f}  {skill_str:>6}  "
            f"{m['mean_pred']:>+9.6f}  {m['std_pred']:>8.6f}"
        )

    b = metrics["_bias"]
    print(f"\n--- Directional Bias (h=1 predictions) ---")
    print(f"  Predicted UP   : {b['pct_positive_pred']:.1%}")
    print(f"  Predicted DOWN : {b['pct_negative_pred']:.1%}")
    print(f"  Mean pred (all): {b['mean_pred_overall']:+.6f}  "
          f"(actual: {b['mean_actual_overall']:+.6f})")

    dir1 = metrics[1]["dir_acc"]
    ic1  = metrics[1]["ic"]
    bias = b["pct_positive_pred"] - 0.5

    print(f"\n--- Interpretation ---")
    if dir1 > 0.55:
        print(f"  [GOOD]  h=1 directional accuracy {dir1:.1%} > 55% — useful signal")
    elif dir1 > 0.52:
        print(f"  [OK]    h=1 directional accuracy {dir1:.1%} — weak but present signal")
    else:
        print(f"  [POOR]  h=1 directional accuracy {dir1:.1%} ≈ random (50% baseline)")

    if abs(ic1) > 0.05:
        print(f"  [GOOD]  h=1 IC {ic1:+.4f} — strong linear correlation with actual returns")
    elif abs(ic1) > 0.02:
        print(f"  [OK]    h=1 IC {ic1:+.4f} — weak but non-zero signal")
    else:
        print(f"  [POOR]  h=1 IC {ic1:+.4f} — near-zero, model has little predictive power")

    if abs(bias) > 0.10:
        direction = "LONG" if bias > 0 else "SHORT"
        print(f"  [BIAS]  Model predicts {direction} {abs(bias):.0%} more than the other "
              f"direction — explains RL agent's {direction}-only behavior")
    else:
        print(f"  [OK]    No strong directional bias in predictions")

    std_ratio = metrics[1]["std_pred"] / max(metrics[1]["std_actual"], 1e-10)
    if std_ratio < 0.1:
        print(f"  [WARN]  Prediction std is only {std_ratio:.1%} of actual std — "
              f"model output is near-constant, directional calls are unreliable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--encoder", default=None,
                        help="Path to encoder .pt. Defaults to config pretrained_encoder.")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override patchtst.horizon from config.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--all-runs", action="store_true",
                        help="Evaluate all PatchTST_N runs and compare side-by-side.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    horizon = args.horizon or cfg["patchtst"].get("horizon", 8)
    window = cfg["env"]["window_size"]

    print("[data] loading market data…")
    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, test_d = split_data(
        data, cfg["data"]["train_split"], cfg["data"]["val_split"]
    )
    train_d, val_d, test_d = standardize_with(train_d, val_d, test_d)
    split_data_map = {"train": train_d, "val": val_d, "test": test_d}
    eval_data = split_data_map[args.split]

    if args.all_runs:
        model_base = Path(cfg["paths"]["model_dir"]) / "patchtst"
        runs = sorted(model_base.glob("PatchTST_*/encoder_best.pt"))
        if not runs:
            print("[error] No PatchTST_N/encoder_best.pt found under", model_base)
            return

        print(f"\nComparing {len(runs)} runs on {args.split} split "
              f"({len(eval_data):,} bars, horizon={horizon})\n")
        print(f"{'Run':<14}  {'h=1 DirAcc':>10}  {'h=1 IC':>8}  "
              f"{'h=4 DirAcc':>10}  {'h=4 IC':>8}  {'Bias UP':>8}  {'PredStd/ActStd':>14}")
        print("-" * 82)
        for enc_path in runs:
            preds, actuals = _collect_predictions(
                enc_path, eval_data, horizon, window, args.batch_size, args.device
            )
            m = _compute_metrics(preds, actuals)
            std_ratio = m[1]["std_pred"] / max(m[1]["std_actual"], 1e-10)
            print(
                f"{enc_path.parent.name:<14}  "
                f"{m[1]['dir_acc']:>10.3f}  {m[1]['ic']:>+8.4f}  "
                f"{m[min(4,horizon)]['dir_acc']:>10.3f}  {m[min(4,horizon)]['ic']:>+8.4f}  "
                f"{m['_bias']['pct_positive_pred']:>8.1%}  {std_ratio:>14.4f}"
            )
        return

    enc_path = args.encoder or cfg["train"].get("pretrained_encoder")
    if not enc_path:
        print("[error] No encoder path. Pass --encoder or set train.pretrained_encoder in config.")
        return

    preds, actuals = _collect_predictions(
        enc_path, eval_data, horizon, window, args.batch_size, args.device
    )
    metrics = _compute_metrics(preds, actuals)
    _print_report(metrics, enc_path, args.split, len(preds))


if __name__ == "__main__":
    main()
