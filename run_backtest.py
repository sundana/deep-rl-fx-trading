"""Run a full backtest with finance metrics + equity curve plot."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.backtest import filter_by_sessions, print_report, run_backtest, save_report
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv
from src.model_utils import load_model


def _latest_run_best_model(model_dir: Path, algo: str) -> Path | None:
    """Return best_model.zip from the most recently created run directory."""
    algo_class = {"PPO": "PPO", "RECURRENTPPO": "RecurrentPPO"}.get(algo.upper(), algo)
    runs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith(f"{algo_class}_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        candidate = run / "best_model.zip"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None,
                        help="Path to model zip. Defaults to best_model of latest run.")
    parser.add_argument("--run", default=None,
                        help="Run name, e.g. RecurrentPPO_4. Uses that run's best_model.zip.")
    parser.add_argument("--algo", choices=["PPO", "RecurrentPPO"], default=None,
                        help="Override train.algo in config.")
    parser.add_argument("--split", choices=["val", "test", "train", "all"], default="test")
    parser.add_argument("--name", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--session", default=None,
        help="Comma-separated sessions to backtest: asia, london, newyork. "
             "E.g. --session asia  or  --session asia,london",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = args.algo or cfg["train"]["algo"]
    model_dir = Path(cfg["paths"]["model_dir"])

    if args.model:
        model_path = args.model
    elif args.run:
        model_path = str(model_dir / args.run / "best_model.zip")
    else:
        found = _latest_run_best_model(model_dir, algo)
        model_path = str(found) if found else cfg["paths"].get("best_model", "models/best_model.zip")

    print(f"[backtest] loading {algo} model from {model_path}  device={args.device}")
    model = load_model(model_path, algo=algo, device=args.device)

    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, test_d = split_data(
        data, cfg["data"]["train_split"], cfg["data"]["val_split"]
    )
    train_d, val_d, test_d = standardize_with(train_d, val_d, test_d)

    if args.split == "all":
        # concatenate val+test for an out-of-sample run
        from src.data_loader import MarketData
        import numpy as np

        selected = MarketData(
            timestamps=np.concatenate([val_d.timestamps, test_d.timestamps]),
            open=np.concatenate([val_d.open, test_d.open]),
            high=np.concatenate([val_d.high, test_d.high]),
            low=np.concatenate([val_d.low, test_d.low]),
            close=np.concatenate([val_d.close, test_d.close]),
            volume=np.concatenate([val_d.volume, test_d.volume]),
            features=np.concatenate([val_d.features, test_d.features], axis=0),
            feature_names=val_d.feature_names,
        )
    else:
        selected = {"train": train_d, "val": val_d, "test": test_d}[args.split]

    sessions = [s.strip() for s in args.session.split(",")] if args.session else None
    if sessions:
        selected = filter_by_sessions(selected, sessions)
        print(f"[backtest] session filter={sessions} bars remaining={len(selected)}")

    print(f"[backtest] split={args.split} bars={len(selected)}")

    env_cfg = EnvConfig(
        window_size=cfg["env"]["window_size"],
        initial_balance=cfg["env"]["initial_balance"],
        transaction_cost_pips=cfg["env"]["transaction_cost_pips"],
        pip_size=cfg["env"]["pip_size"],
        contract_size=cfg["env"]["contract_size"],
        position_size=cfg["env"]["position_size"],
        reward_scaling=cfg["env"]["reward_scaling"],
        drawdown_penalty=0.0,        # no penalty shaping during backtest
        hold_penalty=0.0,
        max_drawdown_pct=0.0,        # never cut episode short in backtest
        trade_penalty=0.0,
        min_hold_bars=0,
        early_exit_penalty=0.0,
        hold_bonus_per_bar=0.0,
        random_start=False,
        episode_length=None,
    )

    env = ForexEnv(selected, env_cfg)
    res = run_backtest(model, env, deterministic=not args.stochastic)
    print_report(res)
    session_tag = ("_" + "_".join(s.lower() for s in sessions)) if sessions else ""
    name = args.name or f"backtest_{args.split}{session_tag}"

    # Save results alongside the model's run directory when possible
    model_p = Path(model_path)
    if model_p.parent.name.startswith(("PPO_", "RecurrentPPO_")):
        out_dir = str(model_p.parent)
    elif args.run:
        out_dir = str(model_dir / args.run)
    else:
        out_dir = cfg["paths"]["results_dir"]

    out = save_report(res, out_dir, name=name)
    print(f"[backtest] saved metrics -> {out}")
    print(f"[backtest] saved chart   -> {out.with_name(name + '_equity.png')}")


if __name__ == "__main__":
    main()
