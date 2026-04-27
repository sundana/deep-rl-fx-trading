"""Gymnasium forex trading environment.

Action space: Discrete(3) — 0 short, 1 flat, 2 long. Each step the agent
chooses a target position; switching positions incurs transaction costs.

Reward: change in log-equity, scaled. Optional drawdown / hold penalties.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_loader import MarketData


SHORT, FLAT, LONG = 0, 1, 2
POS_VALUE = {SHORT: -1, FLAT: 0, LONG: 1}


@dataclass
class EnvConfig:
    window_size: int = 64
    initial_balance: float = 10_000.0
    transaction_cost_pips: float = 1.5
    pip_size: float = 1e-4
    contract_size: float = 10_000.0
    position_size: float = 1.0
    reward_scaling: float = 100.0
    drawdown_penalty: float = 0.0
    hold_penalty: float = 0.0
    random_start: bool = True
    episode_length: int | None = None


class ForexEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, cfg: EnvConfig | None = None):
        super().__init__()
        self.data = data
        self.cfg = cfg or EnvConfig()
        if len(data) <= self.cfg.window_size + 2:
            raise ValueError("Data shorter than window size.")

        self.n_features = data.features.shape[1]
        obs_dim = self.cfg.window_size * self.n_features + 2  # + position + unrealized pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._cost_per_unit = self.cfg.transaction_cost_pips * self.cfg.pip_size
        self._reset_internal()

    def _reset_internal(self) -> None:
        self.balance = self.cfg.initial_balance
        self.equity = self.cfg.initial_balance
        self.peak_equity = self.cfg.initial_balance
        self.position = FLAT
        self.entry_price = 0.0
        self.units = 0.0  # signed units of base currency
        self.t = self.cfg.window_size
        self.end_t = len(self.data) - 1
        self.history: list[dict] = []
        self.trades: list[dict] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_internal()
        if self.cfg.random_start and self.cfg.episode_length:
            max_start = len(self.data) - self.cfg.episode_length - 1
            if max_start > self.cfg.window_size:
                self.t = int(self.np_random.integers(self.cfg.window_size, max_start))
                self.end_t = self.t + self.cfg.episode_length
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        window = self.data.features[self.t - self.cfg.window_size : self.t].ravel()
        price = float(self.data.close[self.t])
        unrealized = 0.0
        if self.position != FLAT and self.entry_price > 0:
            unrealized = self.units * (price - self.entry_price) / self.cfg.initial_balance
        pos_feat = float(POS_VALUE[self.position])
        return np.concatenate(
            [window, np.array([pos_feat, unrealized], dtype=np.float32)]
        ).astype(np.float32)

    def _close_position(self, price: float) -> float:
        if self.position == FLAT:
            return 0.0
        pnl = self.units * (price - self.entry_price)
        cost = abs(self.units) * self._cost_per_unit
        realized = pnl - cost
        self.balance += realized
        self.trades.append(
            {
                "exit_t": self.t,
                "exit_price": price,
                "entry_price": self.entry_price,
                "units": self.units,
                "pnl": realized,
            }
        )
        self.position = FLAT
        self.entry_price = 0.0
        self.units = 0.0
        return realized

    def _open_position(self, target: int, price: float) -> None:
        if target == FLAT:
            return
        notional = self.balance * self.cfg.position_size
        units = notional / price
        units = units if target == LONG else -units
        # Pay opening cost via cost_per_unit on close as well; charge half here, half on close
        self.balance -= abs(units) * self._cost_per_unit * 0.5
        self.position = target
        self.entry_price = price
        self.units = units

    def step(self, action: int):
        action = int(action)
        price = float(self.data.close[self.t])

        prev_equity = self._mark_to_market(price)

        if action != self.position:
            self._close_position(price)
            if action != FLAT:
                self._open_position(action, price)

        # advance time
        self.t += 1
        terminated = False
        truncated = False
        if self.t >= self.end_t:
            # close any open position at final price
            self._close_position(float(self.data.close[self.t]))
            terminated = True

        next_price = float(self.data.close[self.t])
        new_equity = self._mark_to_market(next_price)
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)

        # log-equity reward
        ret = np.log(max(new_equity, 1e-6) / max(prev_equity, 1e-6))
        reward = ret * self.cfg.reward_scaling

        if self.cfg.hold_penalty and self.position == FLAT:
            reward -= self.cfg.hold_penalty
        if self.cfg.drawdown_penalty:
            dd = (self.peak_equity - new_equity) / self.peak_equity
            reward -= self.cfg.drawdown_penalty * dd

        self.history.append(
            {
                "t": self.t,
                "price": next_price,
                "equity": new_equity,
                "balance": self.balance,
                "position": self.position,
                "action": action,
                "reward": reward,
            }
        )

        # bankrupt
        if new_equity <= 0.1 * self.cfg.initial_balance:
            terminated = True

        info = {
            "equity": new_equity,
            "balance": self.balance,
            "position": self.position,
            "n_trades": len(self.trades),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _mark_to_market(self, price: float) -> float:
        if self.position == FLAT:
            return self.balance
        unrealized = self.units * (price - self.entry_price)
        # not subtracting close cost here — cost realized at actual close
        return self.balance + unrealized
