"""Optuna hyperparameter search for PPO and RecurrentPPO.

Usage
-----
# Tune PPO, 50 trials of 200k steps each:
python tune.py --algo PPO --trials 50 --timesteps 200000

# Tune both algos sequentially, persist results to SQLite (resumable):
python tune.py --algo both --trials 50 --study-db sqlite:///results/tuning/tune.db

# View results in browser:
optuna-dashboard sqlite:///results/tuning/tune.db
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.backtest import run_backtest
from src.data_loader import load_market_data, split_data, standardize_with
from src.env import EnvConfig, ForexEnv

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# Number of parallel envs used during each trial (kept fixed to reduce noise)
N_TUNE_ENVS_PPO = 4
N_TUNE_ENVS_RPPO = 2  # bumped from 1; RecurrentPPO handles n_envs > 1 fine


# ---------------------------------------------------------------------------
# Pruning callback
# ---------------------------------------------------------------------------

class TrialEvalCallback(EvalCallback):
    """EvalCallback that reports intermediate values to Optuna for pruning."""

    def __init__(self, eval_env, trial: optuna.Trial, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial
        self._eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self._eval_idx += 1
            self.trial.report(float(self.last_mean_reward), self._eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

_NET_ARCH_MAP = {
    "small":  dict(pi=[128, 128], vf=[128, 128]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
    "large":  dict(pi=[512, 256], vf=[512, 256]),
}


def sample_ppo_params(trial: optuna.Trial, n_envs: int) -> dict[str, Any]:
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    # batch_size must fit within one rollout buffer (n_steps * n_envs)
    max_bs = n_steps * n_envs
    valid_bs = [b for b in [64, 128, 256, 512, 1024] if b <= max_bs]
    batch_size = trial.suggest_categorical("batch_size", valid_bs)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       n_steps,
        "batch_size":    batch_size,
        "n_epochs":      trial.suggest_int("n_epochs", 3, 20),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.90, 1.0),
        "clip_range":    trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "vf_coef":       trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "net_arch":      _NET_ARCH_MAP[
                             trial.suggest_categorical("net_arch", list(_NET_ARCH_MAP))
                         ],
    }


def sample_rppo_params(trial: optuna.Trial, n_envs: int) -> dict[str, Any]:
    params = sample_ppo_params(trial, n_envs)
    params["lstm_hidden_size"] = trial.suggest_categorical(
        "lstm_hidden_size", [64, 128, 256, 512]
    )
    params["n_lstm_layers"] = trial.suggest_int("n_lstm_layers", 1, 2)
    return params


# ---------------------------------------------------------------------------
# Model builder (no tensorboard, verbose=0)
# ---------------------------------------------------------------------------

def _build_model(algo: str, env, params: dict[str, Any], seed: int):
    net_arch = params["net_arch"]
    common: dict[str, Any] = dict(
        env=env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        seed=seed,
        verbose=0,
    )
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=net_arch),
            **common,
        )
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO(
        policy="MlpLstmPolicy",
        policy_kwargs=dict(
            net_arch=net_arch,
            lstm_hidden_size=params["lstm_hidden_size"],
            n_lstm_layers=params["n_lstm_layers"],
            shared_lstm=False,
            enable_critic_lstm=True,
        ),
        **common,
    )


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(
    algo: str,
    train_d,
    val_d,
    env_cfg_dict: dict,
    seed: int,
    n_timesteps: int,
) -> Any:
    n_envs = N_TUNE_ENVS_PPO if algo == "PPO" else N_TUNE_ENVS_RPPO

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        params = (
            sample_ppo_params(trial, n_envs)
            if algo == "PPO"
            else sample_rppo_params(trial, n_envs)
        )

        ep_len = min(8_000, len(train_d) - env_cfg_dict["window_size"] - 2)
        env_cfg = EnvConfig(
            window_size=env_cfg_dict["window_size"],
            initial_balance=env_cfg_dict["initial_balance"],
            transaction_cost_pips=env_cfg_dict["transaction_cost_pips"],
            pip_size=env_cfg_dict["pip_size"],
            contract_size=env_cfg_dict["contract_size"],
            position_size=env_cfg_dict["position_size"],
            reward_scaling=env_cfg_dict["reward_scaling"],
            drawdown_penalty=env_cfg_dict["drawdown_penalty"],
            hold_penalty=env_cfg_dict["hold_penalty"],
            random_start=True,
            episode_length=ep_len,
        )
        eval_cfg = EnvConfig(
            **{**env_cfg.__dict__, "random_start": False, "episode_length": None}
        )

        train_env = DummyVecEnv([
            (lambda _=i: Monitor(ForexEnv(train_d, env_cfg)))
            for i in range(n_envs)
        ])
        eval_env = DummyVecEnv([lambda: Monitor(ForexEnv(val_d, eval_cfg))])

        eval_freq = max(1, n_timesteps // 10 // n_envs)
        eval_cb = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=1,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0,
        )

        try:
            model = _build_model(algo, train_env, params, trial_seed)
            model.learn(n_timesteps, callback=eval_cb, progress_bar=False)
        except (AssertionError, ValueError):
            return float("-inf")
        finally:
            train_env.close()
            eval_env.close()

        if eval_cb.is_pruned:
            raise optuna.exceptions.TrialPruned()

        # Final full-episode backtest on validation set
        result = run_backtest(model, ForexEnv(val_d, eval_cfg), deterministic=True)

        sharpe = result.sharpe if np.isfinite(result.sharpe) else -99.0
        # Composite score: Sharpe is primary, small return bonus breaks ties
        score = sharpe + 0.01 * np.clip(result.total_return, -1.0, 1.0)

        trial.set_user_attr("sharpe", round(float(result.sharpe), 4))
        trial.set_user_attr("total_return", round(float(result.total_return), 4))
        trial.set_user_attr("max_drawdown", round(float(result.max_drawdown), 4))
        trial.set_user_attr("n_trades", int(result.n_trades))

        return float(score)

    return objective


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_study_summary(study: optuna.Study, algo: str) -> None:
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    if not completed:
        print(f"  No completed trials for {algo}.")
        return

    completed.sort(key=lambda t: t.value or float("-inf"), reverse=True)
    best = completed[0]

    print(f"\n{'='*55}")
    print(f" Best {algo} trial  (#{best.number})")
    print(f"{'='*55}")
    print(f"  Score:        {best.value:.4f}")
    print(f"  Sharpe:       {best.user_attrs.get('sharpe', 'n/a')}")
    print(f"  Total return: {best.user_attrs.get('total_return', 'n/a'):.2%}")
    print(f"  Max drawdown: {best.user_attrs.get('max_drawdown', 'n/a'):.2%}")
    print(f"  # trades:     {best.user_attrs.get('n_trades', 'n/a')}")
    print(f"\n  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"\n  Completed: {len(completed)}  Pruned: {pruned}")

    print(f"\n  Top-5 {algo} trials:")
    for rank, t in enumerate(completed[:5], 1):
        ret = t.user_attrs.get("total_return", float("nan"))
        sh  = t.user_attrs.get("sharpe", float("nan"))
        print(
            f"    #{rank}  trial={t.number:<4}  score={t.value:.4f}"
            f"  sharpe={sh:.3f}  return={ret:.2%}"
        )


def _save_best_params(study: optuna.Study, algo: str, out_dir: Path) -> Path:
    best = study.best_trial
    payload = {
        "algo":       algo,
        "score":      best.value,
        "params":     best.params,
        "user_attrs": best.user_attrs,
    }
    path = out_dir / f"best_params_{algo.lower()}.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for PPO / RecurrentPPO"
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument(
        "--algo", choices=["PPO", "RecurrentPPO", "both"], default="PPO",
        help="Which algorithm(s) to tune."
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials per algorithm."
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Training timesteps per trial (shorter = faster, noisier)."
    )
    parser.add_argument(
        "--study-db", default=None,
        help="Optional SQLite URL for persistence, e.g. sqlite:///results/tuning/tune.db"
    )
    parser.add_argument("--out-dir", default="results/tuning")
    parser.add_argument(
        "--n-jobs", type=int, default=min(4, max(1, os.cpu_count() // 5)),
        help="Parallel Optuna trials. Default auto-scales to ~1/5 of CPU cores."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[tune] loading data…")
    data = load_market_data(cfg["data"]["csv_path"], separator=cfg["data"]["separator"])
    train_d, val_d, _ = split_data(
        data, cfg["data"]["train_split"], cfg["data"]["val_split"]
    )
    train_d, val_d = standardize_with(train_d, val_d)
    print(f"[tune] train={len(train_d)} bars  val={len(val_d)} bars")

    algos = ["PPO", "RecurrentPPO"] if args.algo == "both" else [args.algo]

    for algo in algos:
        study_name = f"{algo.lower()}_forex"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.study_db,
            sampler=TPESampler(seed=seed, constant_liar=args.n_jobs > 1),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            direction="maximize",
            load_if_exists=True,
        )

        n_done = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        remaining = args.trials - n_done
        if remaining <= 0:
            print(f"[tune] {algo}: {n_done} trials already complete, skipping.")
            _print_study_summary(study, algo)
            continue

        n_envs = N_TUNE_ENVS_PPO if algo == "PPO" else N_TUNE_ENVS_RPPO
        print(
            f"\n[tune] {algo}: running {remaining} trials "
            f"({args.timesteps:,} steps each, {n_envs} envs, "
            f"{args.n_jobs} parallel)…"
        )

        study.optimize(
            make_objective(
                algo=algo,
                train_d=train_d,
                val_d=val_d,
                env_cfg_dict=cfg["env"],
                seed=seed,
                n_timesteps=args.timesteps,
            ),
            n_trials=remaining,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            catch=(Exception,),
        )

        _print_study_summary(study, algo)

        try:
            path = _save_best_params(study, algo, out_dir)
            print(f"[tune] best params saved → {path}")
        except ValueError:
            print(f"[tune] no completed trials — nothing saved.")

    if args.study_db:
        print(f"\n[tune] to visualize: optuna-dashboard {args.study_db}")


if __name__ == "__main__":
    main()
