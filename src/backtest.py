"""Backtest a trained policy and compute finance metrics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .env import ForexEnv


# 15-minute bars: 96 per day, ~252 trading days/year for FX → annualization factor
BARS_PER_YEAR = 96 * 252

# Forex market session hours in UTC (hour, minute) half-open intervals
_SESSIONS = {
    "Asia":     (0,  9),   # 00:00 – 09:00 UTC
    "London":   (7,  16),  # 07:00 – 16:00 UTC
    "New York": (12, 21),  # 12:00 – 21:00 UTC
}

_SESSION_ALIASES: dict[str, str] = {
    "asia": "Asia",
    "london": "London",
    "newyork": "New York",
    "new_york": "New York",
    "ny": "New York",
}


def _session_of_hour(hour: int) -> list[str]:
    """Return list of sessions active at a given UTC hour (sessions can overlap)."""
    return [name for name, (start, end) in _SESSIONS.items() if start <= hour < end]


def filter_by_sessions(data, sessions: list[str]):
    """Return a MarketData with only bars that fall inside the requested session(s).

    Accepted names (case-insensitive): asia, london, newyork / new_york / ny.
    Sessions that overlap (e.g. London/Asia 07-09) are included when *either*
    requested session is active at that hour.
    """
    import datetime as _dt
    from .data_loader import MarketData

    canonical: set[str] = set()
    for s in sessions:
        key = s.strip().lower()
        if key not in _SESSION_ALIASES:
            raise ValueError(
                f"Unknown session '{s}'. Valid values: "
                + ", ".join(sorted(set(_SESSION_ALIASES.keys())))
            )
        canonical.add(_SESSION_ALIASES[key])

    mask = np.array(
        [
            any(sess in canonical for sess in _session_of_hour(
                _dt.datetime.utcfromtimestamp(int(ts)).hour
            ))
            for ts in data.timestamps
        ],
        dtype=bool,
    )

    return MarketData(
        timestamps=data.timestamps[mask],
        open=data.open[mask],
        high=data.high[mask],
        low=data.low[mask],
        close=data.close[mask],
        volume=data.volume[mask],
        features=data.features[mask],
        feature_names=data.feature_names,
    )


@dataclass
class BacktestResult:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    volatility_annual: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    n_trades: int
    final_equity: float
    initial_equity: float
    bars: int
    equity_curve: np.ndarray
    timestamps: np.ndarray
    trades: list[dict]
    actions: np.ndarray
    action_counts: dict = field(default_factory=dict)      # "SHORT"/"FLAT"/"LONG" -> count
    trades_per_day: dict = field(default_factory=dict)     # "Monday" … "Friday" -> count
    trades_per_session: dict = field(default_factory=dict) # session name -> count

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("equity_curve", "timestamps", "trades", "actions"):
            d.pop(k)
        return d


def run_backtest(model, env: ForexEnv, deterministic: bool = True) -> BacktestResult:
    from .model_utils import PolicyRunner

    runner = PolicyRunner(model, deterministic=deterministic)
    obs, _ = env.reset()
    runner.reset()
    done = False
    actions = []
    while not done:
        action = runner.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        actions.append(action)
        done = terminated or truncated

    history = env.history
    equity = np.array([h["equity"] for h in history], dtype=np.float64)
    ts_idx = np.array([h["t"] for h in history], dtype=np.int64)
    timestamps = env.data.timestamps[ts_idx]

    initial = env.cfg.initial_balance
    final = float(equity[-1]) if len(equity) else initial

    returns = np.diff(np.log(np.clip(equity, 1e-6, None))) if len(equity) > 1 else np.array([0.0])
    bars = len(equity)
    years = bars / BARS_PER_YEAR if bars else 1e-9

    total_return = final / initial - 1.0
    cagr = (final / initial) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    vol_ann = float(returns.std(ddof=1) * np.sqrt(BARS_PER_YEAR)) if len(returns) > 1 else 0.0
    mean_ann = float(returns.mean() * BARS_PER_YEAR) if len(returns) else 0.0
    sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0

    downside = returns[returns < 0]
    dd_std = float(downside.std(ddof=1) * np.sqrt(BARS_PER_YEAR)) if len(downside) > 1 else 0.0
    sortino = mean_ann / dd_std if dd_std > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    trades = env.trades
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64) if trades else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (
        float("inf") if len(wins) else 0.0
    )
    avg_trade = float(pnls.mean()) if len(pnls) else 0.0

    # --- trades per day and per session ---
    import datetime as _dt
    _WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    trades_per_day: dict[str, int] = {d: 0 for d in _WEEKDAYS[:5]}
    trades_per_session: dict[str, int] = {"Asia": 0, "London": 0, "New York": 0}
    for t in trades:
        ts = int(env.data.timestamps[t["exit_t"]])
        dt = _dt.datetime.utcfromtimestamp(ts)
        day_name = _WEEKDAYS[dt.weekday()]
        if day_name in trades_per_day:
            trades_per_day[day_name] += 1
        for session in _session_of_hour(dt.hour):
            trades_per_session[session] += 1

    action_arr = np.array(actions, dtype=np.int64)
    total_actions = len(action_arr) or 1
    action_counts = {
        "SHORT": int(np.sum(action_arr == 0)),
        "FLAT":  int(np.sum(action_arr == 1)),
        "LONG":  int(np.sum(action_arr == 2)),
    }

    return BacktestResult(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_dd),
        volatility_annual=float(vol_ann),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade,
        n_trades=len(trades),
        final_equity=final,
        initial_equity=float(initial),
        bars=bars,
        equity_curve=equity,
        timestamps=timestamps,
        trades=trades,
        actions=action_arr,
        action_counts=action_counts,
        trades_per_day=dict(sorted(trades_per_day.items())),
        trades_per_session=trades_per_session,
    )


def save_report(result: BacktestResult, out_dir: str | Path, name: str = "backtest") -> Path:
    import json

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    metrics_path = out_dir / f"{name}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(result.summary(), f, indent=2)

    # Equity curve plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    dates = result.timestamps.astype("datetime64[s]").astype("datetime64[ns]")
    axes[0].plot(dates, result.equity_curve, label="Equity", color="tab:blue")
    axes[0].set_ylabel("Equity")
    axes[0].set_title(
        f"Backtest — return {result.total_return:.2%} | "
        f"Sharpe {result.sharpe:.2f} | MaxDD {result.max_drawdown:.2%} | "
        f"Trades {result.n_trades}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    peak = np.maximum.accumulate(result.equity_curve)
    dd = (result.equity_curve - peak) / peak
    axes[1].fill_between(dates, dd, 0, color="tab:red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"{name}_equity.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)

    return metrics_path


def print_report(result: BacktestResult) -> None:
    s = result.summary()
    print("\n=== Backtest Report ===")
    print(f"Bars:            {s['bars']}")
    print(f"Initial equity:  {s['initial_equity']:.2f}")
    print(f"Final equity:    {s['final_equity']:.2f}")
    print(f"Total return:    {s['total_return']:.2%}")
    print(f"CAGR:            {s['cagr']:.2%}")
    print(f"Volatility (ann):{s['volatility_annual']:.2%}")
    print(f"Sharpe:          {s['sharpe']:.3f}")
    print(f"Sortino:         {s['sortino']:.3f}")
    print(f"Calmar:          {s['calmar']:.3f}")
    print(f"Max drawdown:    {s['max_drawdown']:.2%}")
    print(f"Win rate:        {s['win_rate']:.2%}")
    print(f"Profit factor:   {s['profit_factor']:.3f}")
    print(f"Avg trade PnL:   {s['avg_trade_pnl']:.4f}")
    print(f"# trades:        {s['n_trades']}")

    print("\n--- Action Distribution ---")
    total_actions = sum(result.action_counts.values()) or 1
    for action_name, count in result.action_counts.items():
        pct = count / total_actions
        print(f"  {action_name:<6}: {count:>6}  ({pct:.1%})")

    print("\n--- Trades per Day of Week ---")
    if result.n_trades:
        for day, count in result.trades_per_day.items():
            pct = count / result.n_trades if result.n_trades else 0.0
            print(f"  {day:<10}: {count:>4}  ({pct:.1%})")
    else:
        print("  No trades.")

    print("\n--- Trades per Market Session (UTC) ---")
    session_ranges = {"Asia": "00–09", "London": "07–16", "New York": "12–21"}
    for session, count in result.trades_per_session.items():
        pct = count / result.n_trades if result.n_trades else 0.0
        print(f"  {session:<10} [{session_ranges[session]} UTC]: {count:>4}  ({pct:.1%})")
    note = "(sessions overlap: London/Asia 07–09, London/NY 12–16)"
    print(f"  {note}")
