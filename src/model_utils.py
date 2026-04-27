"""Utilities for loading and using PPO / RecurrentPPO models interchangeably."""
from __future__ import annotations

from typing import Any

import numpy as np


def load_model(path: str, algo: str = "PPO"):
    """Load a saved model; detects class from algo name."""
    algo = algo.upper()
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(path)
    elif algo == "RECURRENTPPO":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path)
    raise ValueError(f"Unknown algo '{algo}'. Choose 'PPO' or 'RecurrentPPO'.")


def infer_algo(model) -> str:
    cls = type(model).__name__
    if cls == "PPO":
        return "PPO"
    if cls == "RecurrentPPO":
        return "RecurrentPPO"
    raise TypeError(f"Unsupported model type: {cls}")


class PolicyRunner:
    """Wraps a PPO or RecurrentPPO model with a unified step() interface."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic
        self._is_recurrent = infer_algo(model) == "RecurrentPPO"
        self.lstm_states: Any = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def reset(self):
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def predict(self, obs: np.ndarray) -> int:
        obs_in = obs[np.newaxis]  # (1, obs_dim)
        if self._is_recurrent:
            action, self.lstm_states = self.model.predict(
                obs_in,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=self.deterministic,
            )
            self.episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(obs_in, deterministic=self.deterministic)
        return int(action[0])


def build_model(algo: str, env, cfg: dict[str, Any], log_dir: str):
    """Instantiate a fresh PPO or RecurrentPPO from config."""
    common: dict[str, Any] = dict(
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        tensorboard_log=log_dir,
        seed=cfg["seed"],
        device=cfg["device"],
        verbose=1,
    )
    if algo.upper() == "PPO":
        from stable_baselines3 import PPO
        net_arch = cfg.get("net_arch", dict(pi=[256, 256], vf=[256, 256]))
        return PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=net_arch),
            **common,
        )
    elif algo.upper() == "RECURRENTPPO":
        from sb3_contrib import RecurrentPPO
        net_arch = cfg.get("net_arch", dict(pi=[256], vf=[256]))
        return RecurrentPPO(
            policy="MlpLstmPolicy",
            policy_kwargs=dict(
                net_arch=net_arch,
                lstm_hidden_size=cfg.get("lstm_hidden_size", 256),
                n_lstm_layers=cfg.get("n_lstm_layers", 1),
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            **common,
        )
    raise ValueError(f"Unknown algo '{algo}'.")
