"""UTC-hour-based filtering of MarketData to specific FX trading sessions.

Sessions and their UTC hours: asia=00-09, london=07-16, newyork=12-21.
Overlap bars (e.g. 07-09 is both Asia and London) are included if either
session is selected.

NOTE: after filtering, consecutive bars in the returned dataset are NOT
necessarily 15-minute apart — there will be time gaps at session boundaries.
ForexEnv treats them as sequential, so positions held across session
boundaries span those gaps silently.
"""
from __future__ import annotations

import numpy as np

from .data_loader import MarketData


VALID_SESSIONS = {"asia", "london", "newyork"}
SESSION_HOURS = {
    "asia":    (0,  9),
    "london":  (7,  16),
    "newyork": (12, 21),
}


def filter_data_by_sessions(data: MarketData, sessions: list[str]) -> MarketData:
    sessions_set = {s.lower() for s in sessions}
    unknown = sessions_set - VALID_SESSIONS
    if unknown:
        raise ValueError(
            f"Unknown session(s): {unknown}. Valid options: {VALID_SESSIONS}"
        )

    hours = (data.timestamps % 86400) // 3600  # vectorised UTC-hour extraction
    mask = np.zeros(len(data.timestamps), dtype=bool)
    for session in sessions_set:
        start, end = SESSION_HOURS[session]
        mask |= (hours >= start) & (hours < end)

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
