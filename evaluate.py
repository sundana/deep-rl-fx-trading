"""Evaluate a trained model on val/test split: mean reward + quick metrics."""
from __future__ import annotations

import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO

from src.backtest import print_report, run_backtest
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Defaults to paths.best_model.")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model or cfg["paths"]["best_model"]
    print(f"[eval] loading model {model_path}")
    model = PPO.load(model_path)

    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, test_d = split_data(
        data, cfg["data"]["train_split"], cfg["data"]["val_split"]
    )
    train_d, val_d, test_d = standardize_with(train_d, val_d, test_d)
    selected = {"train": train_d, "val": val_d, "test": test_d}[args.split]
    print(f"[eval] split={args.split} bars={len(selected)}")

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

    rewards, returns_, sharpes = [], [], []
    last_res = None
    for ep in range(args.episodes):
        env = ForexEnv(selected, env_cfg)
        res = run_backtest(model, env, deterministic=not args.stochastic)
        last_res = res
        rewards.append(sum(h["reward"] for h in env.history))
        returns_.append(res.total_return)
        sharpes.append(res.sharpe)
        print(f"[eval] ep {ep + 1}: reward={rewards[-1]:.2f}  return={res.total_return:.2%}  sharpe={res.sharpe:.2f}")

    print(
        f"\n[eval] mean reward={np.mean(rewards):.2f} ± {np.std(rewards):.2f} | "
        f"mean return={np.mean(returns_):.2%} | mean sharpe={np.mean(sharpes):.2f}"
    )
    if last_res is not None:
        print_report(last_res)


if __name__ == "__main__":
    main()
