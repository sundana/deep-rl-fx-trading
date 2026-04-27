"""Train an SB3 PPO agent on the EURUSD M15 forex environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.backtest import print_report, run_backtest, save_report
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv


def make_env_factory(data, env_cfg: EnvConfig, seed: int = 0):
    def _thunk():
        env = ForexEnv(data, env_cfg)
        env.reset(seed=seed)
        return Monitor(env)

    return _thunk


def build_vec_env(data, env_cfg, n_envs: int, seed: int, subproc: bool = False):
    factories = [make_env_factory(data, env_cfg, seed + i) for i in range(n_envs)]
    cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    return cls(factories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("[data] loading market data…")
    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, _ = split_data(
        data, cfg["data"]["train_split"], cfg["data"]["val_split"]
    )
    train_d, val_d = standardize_with(train_d, val_d)
    print(f"[data] train bars={len(train_d)}  val bars={len(val_d)}  features={train_d.features.shape[1]}")

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
        random_start=True,
        episode_length=min(20_000, len(train_d) - cfg["env"]["window_size"] - 2),
    )

    n_envs = args.n_envs or cfg["train"]["n_envs"]
    train_env = build_vec_env(train_d, env_cfg, n_envs=n_envs, seed=seed, subproc=args.subproc)

    eval_env_cfg = EnvConfig(**{**env_cfg.__dict__, "random_start": False, "episode_length": None})
    eval_env = DummyVecEnv([make_env_factory(val_d, eval_env_cfg, seed=seed + 999)])

    model_dir = Path(cfg["paths"]["model_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        print(f"[train] resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
        model = PPO(
            policy=cfg["train"]["policy"],
            env=train_env,
            learning_rate=cfg["train"]["learning_rate"],
            n_steps=cfg["train"]["n_steps"],
            batch_size=cfg["train"]["batch_size"],
            n_epochs=cfg["train"]["n_epochs"],
            gamma=cfg["train"]["gamma"],
            gae_lambda=cfg["train"]["gae_lambda"],
            clip_range=cfg["train"]["clip_range"],
            ent_coef=cfg["train"]["ent_coef"],
            vf_coef=cfg["train"]["vf_coef"],
            max_grad_norm=cfg["train"]["max_grad_norm"],
            tensorboard_log=str(log_dir),
            seed=seed,
            verbose=1,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, cfg["train"]["save_freq"] // n_envs),
            save_path=str(model_dir / "checkpoints"),
            name_prefix="ppo_forex",
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
    print(f"[train] starting PPO for {total} timesteps on {n_envs} envs")
    model.learn(total_timesteps=total, callback=callbacks, progress_bar=True)
    model.save(cfg["paths"]["final_model"])
    print(f"[train] saved final model -> {cfg['paths']['final_model']}")

    print("[eval] quick val backtest…")
    eval_native = ForexEnv(val_d, eval_env_cfg)
    res = run_backtest(model, eval_native, deterministic=True)
    print_report(res)
    save_report(res, cfg["paths"]["results_dir"], name="val_after_train")


if __name__ == "__main__":
    main()
