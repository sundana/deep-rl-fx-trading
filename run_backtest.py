"""Run a full backtest with finance metrics + equity curve plot."""
from __future__ import annotations

import argparse

import yaml

from src.backtest import print_report, run_backtest, save_report
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv
from src.model_utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Defaults to paths.best_model.")
    parser.add_argument("--algo", choices=["PPO", "RecurrentPPO"], default=None,
                        help="Override train.algo in config.")
    parser.add_argument("--split", choices=["val", "test", "train", "all"], default="test")
    parser.add_argument("--name", default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = args.algo or cfg["train"]["algo"]
    model_path = args.model or cfg["paths"]["best_model"]
    print(f"[backtest] loading {algo} model from {model_path}")
    model = load_model(model_path, algo=algo)

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

    print(f"[backtest] split={args.split} bars={len(selected)}")

    env_cfg = EnvConfig(
        window_size=cfg["env"]["window_size"],
        initial_balance=cfg["env"]["initial_balance"],
        transaction_cost_pips=cfg["env"]["transaction_cost_pips"],
        pip_size=cfg["env"]["pip_size"],
        contract_size=cfg["env"]["contract_size"],
        position_size=cfg["env"]["position_size"],
        reward_scaling=cfg["env"]["reward_scaling"],
        drawdown_penalty=cfg["env"]["drawdown_penalty"],
        hold_penalty=cfg["env"]["hold_penalty"],
        random_start=False,
        episode_length=None,
    )

    env = ForexEnv(selected, env_cfg)
    res = run_backtest(model, env, deterministic=not args.stochastic)
    print_report(res)
    name = args.name or f"backtest_{args.split}"
    out = save_report(res, cfg["paths"]["results_dir"], name=name)
    print(f"[backtest] saved metrics -> {out}")
    print(f"[backtest] saved chart   -> {out.with_name(name + '_equity.png')}")


if __name__ == "__main__":
    main()
