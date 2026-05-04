"""Utilities for loading and using PPO / RecurrentPPO models interchangeably."""
from __future__ import annotations

from typing import Any

import numpy as np


def load_model(path: str, algo: str = "PPO", device: str = "cpu"):
    """Load a saved model; detects class from algo name."""
    algo = algo.upper()
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(path, device=device)
    elif algo == "RECURRENTPPO":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device=device)
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


_PATCHTST_ENCODER_KEYS = (
    "window", "n_features", "patch_len", "stride", "d_model",
    "n_heads", "depth", "dropout", "head_dim", "embed_dim",
)


def _build_patchtst_policy_kwargs(
    net_arch, train_cfg: dict[str, Any], full_cfg: dict[str, Any] | None
) -> dict[str, Any]:
    if not full_cfg or "patchtst" not in full_cfg:
        raise ValueError(
            "feature_extractor=patchtst requires a 'patchtst' section in the full config."
        )
    from src.patchtst import PatchTSTFeaturesExtractor

    src = full_cfg["patchtst"]
    cfg_dict = {k: src[k] for k in _PATCHTST_ENCODER_KEYS if k in src}
    cfg_dict.setdefault("window", full_cfg["env"]["window_size"])
    if "n_features" not in cfg_dict:
        raise ValueError(
            "patchtst.n_features must be set (inject from data.features.shape[1] in caller)."
        )
    return dict(
        net_arch=net_arch,
        features_extractor_class=PatchTSTFeaturesExtractor,
        features_extractor_kwargs=dict(
            patchtst_cfg_dict=cfg_dict,
            pretrained_path=train_cfg.get("pretrained_encoder"),
            freeze=bool(train_cfg.get("freeze_encoder", False)),
        ),
    )


def build_model(
    algo: str,
    env,
    cfg: dict[str, Any],
    log_dir: str,
    full_cfg: dict[str, Any] | None = None,
):
    """Instantiate a fresh PPO or RecurrentPPO from config.

    `full_cfg` is the full yaml dict (for sections beyond `train`, e.g. `patchtst`).
    Optional for backwards compatibility — only required when using a non-MLP
    feature extractor.
    """
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
    feature_extractor = (cfg.get("feature_extractor") or "mlp").lower()

    if algo.upper() == "PPO":
        from stable_baselines3 import PPO
        net_arch = cfg.get("net_arch", dict(pi=[256, 256], vf=[256, 256]))
        if feature_extractor == "mlp":
            policy_kwargs = dict(net_arch=net_arch)
        elif feature_extractor == "patchtst":
            policy_kwargs = _build_patchtst_policy_kwargs(net_arch, cfg, full_cfg)
        else:
            raise ValueError(f"Unknown feature_extractor: {feature_extractor!r}")
        return PPO(
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            **common,
        )
    elif algo.upper() == "RECURRENTPPO":
        if feature_extractor != "mlp":
            raise ValueError(
                f"feature_extractor={feature_extractor!r} not supported with RecurrentPPO. "
                "Use algo=PPO instead."
            )
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
