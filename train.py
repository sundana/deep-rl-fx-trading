"""Train PPO or RecurrentPPO on the EURUSD M15 forex environment."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.backtest import print_report, run_backtest, save_report
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv
from src.model_utils import build_model, load_model


def make_env_factory(data, env_cfg: EnvConfig, seed: int = 0):
    def _thunk():
        env = ForexEnv(data, env_cfg)
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(data, env_cfg, n_envs: int, seed: int, subproc: bool = False):
    factories: list = [make_env_factory(data, env_cfg, seed + i) for i in range(n_envs)]
    cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    return cls(list(factories))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--algo", choices=["PPO", "RecurrentPPO"], default=None,
                        help="Override train.algo in config.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    algo = (args.algo or cfg["train"]["algo"]).upper()
    seed = cfg["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[train] algorithm: {algo}")
    print("[data] loading market data…")
    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, _ = split_data(data, cfg["data"]["train_split"], cfg["data"]["val_split"])
    train_d, val_d = standardize_with(train_d, val_d)
    print(f"[data] train={len(train_d)} bars  val={len(val_d)} bars  features={train_d.features.shape[1]}")

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
        max_drawdown_pct=cfg["env"].get("max_drawdown_pct", 0.0),
        trade_penalty=cfg["env"].get("trade_penalty", 0.0),
        min_hold_bars=cfg["env"].get("min_hold_bars", 0),
        early_exit_penalty=cfg["env"].get("early_exit_penalty", 0.5),
        hold_bonus_per_bar=cfg["env"].get("hold_bonus_per_bar", 0.0),
        random_start=True,
        episode_length=min(20_000, len(train_d) - cfg["env"]["window_size"] - 2),
    )
    eval_env_cfg = EnvConfig(**{
        **env_cfg.__dict__,
        "random_start": False,
        "episode_length": None,
        # disable all training-specific shaping during evaluation
        "drawdown_penalty": 0.0,
        "max_drawdown_pct": 0.0,
        "trade_penalty": 0.0,
        "min_hold_bars": 0,
        "early_exit_penalty": 0.0,
        "hold_bonus_per_bar": 0.0,
    })

    # RecurrentPPO only supports n_envs=1 in SubprocVecEnv due to LSTM state handling;
    # DummyVecEnv works fine for any n_envs.
    n_envs = args.n_envs or cfg["train"]["n_envs"]
    use_subproc = args.subproc and algo != "RECURRENTPPO"
    train_env = build_vec_env(train_d, env_cfg, n_envs=n_envs, seed=seed, subproc=use_subproc)
    eval_env = DummyVecEnv([make_env_factory(val_d, eval_env_cfg, seed=seed + 999)])

    model_dir = Path(cfg["paths"]["model_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    prefix = algo.lower()
    checkpoint_dir = model_dir / "checkpoints"

    if args.resume and Path(args.resume).exists():
        print(f"[train] resuming from {args.resume}")
        model = load_model(args.resume, algo=algo)
        model.set_env(train_env)
    else:
        model = build_model(algo, train_env, cfg["train"], str(log_dir))

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, cfg["train"]["save_freq"] // n_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"{prefix}_forex",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=max(1, cfg["train"]["eval_freq"] // n_envs),
            n_eval_episodes=1,
            deterministic=True,
            render=False,
        ),
    ]

    total = args.timesteps or cfg["train"]["total_timesteps"]
    print(f"[train] starting {algo} for {total} timesteps on {n_envs} envs")
    model.learn(total_timesteps=total, callback=callbacks, progress_bar=True)
    model.save(cfg["paths"]["final_model"])
    print(f"[train] saved final model -> {cfg['paths']['final_model']}")

    print("[eval] quick val backtest…")
    eval_native = ForexEnv(val_d, eval_env_cfg)
    res = run_backtest(model, eval_native, deterministic=True)
    print_report(res)
    save_report(res, cfg["paths"]["results_dir"], name=f"val_{prefix}_after_train")


if __name__ == "__main__":
    main()
