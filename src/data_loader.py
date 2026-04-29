"""Polars-based OHLC loader with technical features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


FEATURE_COLUMNS = [
    "log_return",
    "ret_5",
    "ret_15",
    "ret_60",
    "rsi_14",
    "ema_fast_dev",
    "ema_slow_dev",
    "ema_200_dev",    # long-term trend context (~weekly on M15)
    "atr_14_norm",
    "bb_pct",
    "macd_hist",      # trend momentum acceleration
    "stoch_k",        # range-based overbought/oversold
    "vol_ratio",      # volume conviction
    "hl_range",
    "oc_range",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


@dataclass
class MarketData:
    timestamps: np.ndarray  # int64 unix seconds
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    features: np.ndarray  # (T, F) standardized
    feature_names: list[str]

    def __len__(self) -> int:
        return self.close.shape[0]

    def slice(self, start: int, end: int) -> "MarketData":
        return MarketData(
            timestamps=self.timestamps[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
            features=self.features[start:end],
            feature_names=self.feature_names,
        )


def _load_raw(path: str | Path, separator: str = "\t") -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator=separator,
        has_header=False,
        new_columns=["datetime", "open", "high", "low", "close", "volume"],
        try_parse_dates=False,
    )
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False)
    ).sort("datetime")
    return df


def _add_features(df: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")
    open_ = pl.col("open")

    # Log returns
    log_ret = (close / close.shift(1)).log()

    # RSI(14) using Wilder smoothing approximation via EWM
    delta = close.diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1 / 14, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1 / 14, adjust=False)
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # EMA deviations
    ema_fast = close.ewm_mean(span=12, adjust=False)
    ema_slow = close.ewm_mean(span=48, adjust=False)
    ema_200 = close.ewm_mean(span=200, adjust=False)

    # ATR(14) as % of price
    tr = pl.max_horizontal(
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    )
    atr = tr.ewm_mean(alpha=1 / 14, adjust=False)

    # Bollinger %B
    ma20 = close.rolling_mean(window_size=20)
    sd20 = close.rolling_std(window_size=20)
    bb_pct = (close - ma20) / (4.0 * sd20 + 1e-12)

    # MACD histogram (trend momentum acceleration)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm_mean(span=9, adjust=False)
    macd_hist = (macd_line - macd_signal) / (close + 1e-12)

    # Stochastic %K (range-based overbought/oversold, centered at 0)
    low_14 = low.rolling_min(window_size=14)
    high_14 = high.rolling_max(window_size=14)
    stoch_k = (close - low_14) / (high_14 - low_14 + 1e-12) - 0.5

    # Volume ratio: log deviation from 20-bar mean (conviction of price moves)
    vol_ma = pl.col("volume").rolling_mean(window_size=20)
    vol_ratio = (pl.col("volume") / (vol_ma + 1e-12)).log()

    df = df.with_columns(
        log_ret.alias("log_return"),
        log_ret.rolling_sum(window_size=5).alias("ret_5"),
        log_ret.rolling_sum(window_size=15).alias("ret_15"),
        log_ret.rolling_sum(window_size=60).alias("ret_60"),
        rsi.alias("rsi_14"),
        ((close - ema_fast) / close).alias("ema_fast_dev"),
        ((close - ema_slow) / close).alias("ema_slow_dev"),
        ((close - ema_200) / close).alias("ema_200_dev"),
        (atr / close).alias("atr_14_norm"),
        bb_pct.alias("bb_pct"),
        macd_hist.alias("macd_hist"),
        stoch_k.alias("stoch_k"),
        vol_ratio.alias("vol_ratio"),
        ((high - low) / close).alias("hl_range"),
        ((close - open_) / close).alias("oc_range"),
    )

    df = df.with_columns(
        (pl.col("datetime").dt.hour().cast(pl.Float64) * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (pl.col("datetime").dt.hour().cast(pl.Float64) * (2 * np.pi / 24)).cos().alias("hour_cos"),
        (pl.col("datetime").dt.weekday().cast(pl.Float64) * (2 * np.pi / 7)).sin().alias("dow_sin"),
        (pl.col("datetime").dt.weekday().cast(pl.Float64) * (2 * np.pi / 7)).cos().alias("dow_cos"),
    )

    # Normalize RSI to roughly zero-mean
    df = df.with_columns(((pl.col("rsi_14") - 50.0) / 50.0).alias("rsi_14"))
    return df


def load_market_data(
    path: str | Path,
    separator: str = "\t",
    feature_columns: list[str] | None = None,
) -> MarketData:
    df = _load_raw(path, separator=separator)
    df = _add_features(df)
    df = df.drop_nulls().with_columns(
        [pl.col(c).fill_nan(0.0) for c in df.columns if df.schema[c] == pl.Float64]
    )

    cols = feature_columns or FEATURE_COLUMNS
    feats = df.select(cols).to_numpy().astype(np.float32)

    # Clip extreme outliers, then z-score using training-side robust stats.
    # The env will further use raw values; standardization here keeps the
    # observation roughly in [-3, 3].
    feats = np.clip(feats, -10.0, 10.0)

    timestamps = df.get_column("datetime").to_numpy().astype("datetime64[s]").astype(np.int64)

    return MarketData(
        timestamps=timestamps,
        open=df.get_column("open").to_numpy().astype(np.float32),
        high=df.get_column("high").to_numpy().astype(np.float32),
        low=df.get_column("low").to_numpy().astype(np.float32),
        close=df.get_column("close").to_numpy().astype(np.float32),
        volume=df.get_column("volume").to_numpy().astype(np.float32),
        features=feats,
        feature_names=cols,
    )


def standardize_with(train: MarketData, *others: MarketData) -> tuple[MarketData, ...]:
    """Z-score features using stats from the first half of `train` only (no leakage)."""
    n_burnin = max(1, len(train.features) // 2)
    mu = train.features[:n_burnin].mean(axis=0, keepdims=True)
    sd = train.features[:n_burnin].std(axis=0, keepdims=True) + 1e-8

    def _apply(d: MarketData) -> MarketData:
        return MarketData(
            timestamps=d.timestamps,
            open=d.open,
            high=d.high,
            low=d.low,
            close=d.close,
            volume=d.volume,
            features=((d.features - mu) / sd).astype(np.float32),
            feature_names=d.feature_names,
        )

    return tuple(_apply(d) for d in (train, *others))


def split_data(
    data: MarketData, train_frac: float, val_frac: float
) -> tuple[MarketData, MarketData, MarketData]:
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data.slice(0, n_train)
    val = data.slice(n_train, n_train + n_val)
    test = data.slice(n_train + n_val, n)
    return train, val, test
