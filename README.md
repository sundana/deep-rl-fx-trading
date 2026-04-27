# RL Forex Trading Robot — EURUSD M15

Deep reinforcement learning trading bot for EURUSD on the 15-minute timeframe,
built on **Stable-Baselines3 (PPO)**, **Gymnasium**, and **Polars**.

## Layout

```
data/EURUSD15.csv           # OHLC + volume, tab-separated, no header
config.yaml                 # all hyperparameters
src/
  data_loader.py            # polars OHLC loader + technical features
  env.py                    # Gymnasium env: long/flat/short, cost-aware
  backtest.py               # finance metrics + equity-curve plot
train.py                    # PPO training (TensorBoard + EvalCallback)
evaluate.py                 # quick eval on val/test/train
run_backtest.py             # full backtest with report + chart
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
# PPO (default)
python train.py
python train.py --timesteps 200000 --n-envs 4
python train.py --resume models/checkpoints/ppo_forex_500000_steps.zip

# RecurrentPPO (MlpLstmPolicy — override via CLI or set algo: RecurrentPPO in config.yaml)
python train.py --algo RecurrentPPO
python train.py --algo RecurrentPPO --timesteps 500000

tensorboard --logdir logs
```

The training script:
1. Loads CSV with Polars and computes features (log returns over multiple horizons,
   RSI(14), EMA deviations, ATR(14)/price, Bollinger %B, hour/dow encodings).
2. Splits 70/15/15 train/val/test, z-scores features using **train stats only**.
3. Vectorizes the env (DummyVecEnv or SubprocVecEnv with `--subproc`).
4. Trains PPO with periodic eval on the val split; best model saved to
   `models/best_model.zip`.

## Evaluate

```bash
python evaluate.py --split val
python evaluate.py --split test --episodes 5
python evaluate.py --algo RecurrentPPO --split test
```

## Backtest

```bash
python run_backtest.py --split test
python run_backtest.py --split all --name oos_full
python run_backtest.py --algo RecurrentPPO --split test
```

Produces `results/<name>_metrics.json` and `results/<name>_equity.png` with:

- Total return, CAGR, annualized volatility
- Sharpe, Sortino, Calmar ratios
- Max drawdown
- Win rate, profit factor, average trade PnL
- Number of trades, equity curve, drawdown curve

Annualization assumes 96 bars/day × 252 days = 24,192 bars/year.

## Environment

- **Observation**: window of 64 bars × 15 standardized features, plus current
  position and unrealized PnL → vector of shape `(64*15 + 2,)`.
- **Action**: `Discrete(3)` — 0 short, 1 flat, 2 long. Each step the agent picks
  a target position; switching closes the old one and opens the new one.
- **Reward**: change in log-equity × `reward_scaling`, with optional drawdown
  and hold penalties (off by default).
- **Costs**: `transaction_cost_pips` × `pip_size` per unit, charged half on open
  / half on close (round-trip = full spread).

## Why these choices

- **PPO**: robust default for discrete control, stable on noisy financial
  rewards, and well-supported in SB3.
- **RecurrentPPO** (`MlpLstmPolicy` from `sb3-contrib`): adds a shared LSTM
  layer so the agent can maintain hidden state across steps — useful for
  detecting trend/regime patterns that span more than the observation window.
  Typically converges slower but can learn temporal dependencies PPO+MLP
  cannot. Switch with `--algo RecurrentPPO` or `algo: RecurrentPPO` in
  `config.yaml`; tune `lstm_hidden_size` and `n_lstm_layers` there.
- **Discrete long/flat/short**: cleanest signal-to-action mapping, avoids
  exploding the action space with sizing decisions the value function can't yet
  ground in P&L.
- **Log-equity reward**: scale-invariant, keeps gradients sane across regimes.
- **Polars**: features computed lazily over a million-row table in one pass —
  much faster than equivalent Pandas/NumPy pipelines, and the schema-strict API
  catches bugs at load time.
- **Train-only normalization**: feature stats are computed on the train split
  and applied to val/test to avoid look-ahead leakage.

## Notes

- The loader expects a tab-separated file: `datetime  O  H  L  C  V`. Adjust
  `data.separator` in `config.yaml` for comma-separated dumps.
- 1M timesteps is a starting point. Strong policies on M15 typically need
  3–10M timesteps and curriculum tuning of `transaction_cost_pips`,
  `position_size`, and `episode_length`.
- Out-of-sample performance is what matters. Always read `results/backtest_test_metrics.json`
  before trusting a model.
