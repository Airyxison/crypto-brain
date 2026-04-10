"""
Feature Engineering
-------------------
Converts raw tick data into a normalized 13-dimensional state vector.
Operates on a rolling window — no lookahead, no leakage.

Features:
  [0]  price_norm          — price normalized to recent range
  [1]  momentum_1m         — 1-min return
  [2]  momentum_5m         — 5-min return
  [3]  momentum_15m        — 15-min return
  [4]  volatility_1h       — 1-hour realized volatility (rolling std of returns)
  [5]  volume_norm         — volume normalized to recent average
  [6]  vwap_deviation      — (price - vwap) / vwap
  [7]  in_position         — 0.0 or 1.0
  [8]  position_pnl        — unrealized P&L as fraction (-1 to 1 clipped)
  [9]  time_in_position    — bars held, normalized (0-1 over max_hold window)
  [10] distance_to_stop    — (price - stop) / price, 0 if no position
  [11] trade_frequency     — ticks per minute normalized to recent avg
  [12] price_vs_range      — (price - 4h_low) / (4h_high - 4h_low)
"""

import numpy as np
from collections import deque


# Tick window requirements
WINDOW_1M   = 60       # ~60 ticks/min at 1Hz
WINDOW_5M   = 300
WINDOW_15M  = 900
WINDOW_1H   = 3600
WINDOW_4H   = 14400
MIN_WINDOW  = WINDOW_5M   # 5 min of history to start — lower bar for POC


class FeatureEngineer:
    def __init__(self):
        self.prices     = deque(maxlen=WINDOW_4H)
        self.volumes    = deque(maxlen=WINDOW_4H)
        self.times      = deque(maxlen=WINDOW_4H)  # trade_time in ms
        self._ready     = False

    def update(self, price: float, volume: float, trade_time: int):
        self.prices.append(price)
        self.volumes.append(volume)
        self.times.append(trade_time)
        if len(self.prices) >= MIN_WINDOW:
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def extract(self, position_state: dict) -> np.ndarray:
        """
        position_state = {
            'in_position': bool,
            'entry_price': float | None,
            'stop_price':  float | None,
            'bars_held':   int,
            'max_hold':    int,
        }
        """
        if not self._ready:
            raise RuntimeError("Not enough data — call update() at least MIN_WINDOW times")

        prices  = np.array(self.prices,  dtype=np.float64)
        volumes = np.array(self.volumes, dtype=np.float64)
        current = prices[-1]

        # --- Price features ---
        def _momentum(n: int) -> float:
            window = min(n, len(prices))
            base = prices[-window]
            return (current - base) / base if base != 0 else 0.0

        # Normalized price: position within 4h range
        lo4h, hi4h = prices.min(), prices.max()
        price_vs_range = (current - lo4h) / (hi4h - lo4h + 1e-9)

        # Rolling 1h volatility (std of 1-step returns)
        n1h = min(WINDOW_1H, len(prices))
        returns_1h = np.diff(prices[-n1h:]) / prices[-n1h:-1]
        volatility_1h = float(returns_1h.std()) if len(returns_1h) > 1 else 0.0

        # --- Volume features ---
        vol_avg = volumes[-WINDOW_5M:].mean() if len(volumes) >= WINDOW_5M else volumes.mean()
        volume_norm = float(volumes[-1] / (vol_avg + 1e-9)) - 1.0  # centered at 0

        # VWAP over 1h window
        n_vwap = min(WINDOW_1H, len(prices))
        vwap = float((prices[-n_vwap:] * volumes[-n_vwap:]).sum() / (volumes[-n_vwap:].sum() + 1e-9))
        vwap_deviation = (current - vwap) / (vwap + 1e-9)

        # --- Trade frequency ---
        if len(self.times) >= WINDOW_5M:
            elapsed_s = (self.times[-1] - self.times[-WINDOW_5M]) / 1000.0
            freq_now = WINDOW_5M / (elapsed_s + 1e-9)
        else:
            freq_now = len(self.times)

        # Normalize frequency: compare to 1h avg
        if len(self.times) >= WINDOW_1H:
            elapsed_1h = (self.times[-1] - self.times[-WINDOW_1H]) / 1000.0
            freq_1h = WINDOW_1H / (elapsed_1h + 1e-9)
        else:
            freq_1h = freq_now
        trade_frequency = (freq_now / (freq_1h + 1e-9)) - 1.0

        # --- Position features ---
        in_pos = float(position_state.get('in_position', False))
        entry  = position_state.get('entry_price')
        stop   = position_state.get('stop_price')
        bars   = position_state.get('bars_held', 0)
        max_h  = position_state.get('max_hold', 1440)

        position_pnl = 0.0
        if in_pos and entry:
            position_pnl = np.clip((current - entry) / (entry + 1e-9), -1.0, 1.0)

        time_in_pos = np.clip(bars / (max_h + 1e-9), 0.0, 1.0)

        dist_to_stop = 0.0
        if in_pos and stop and current > 0:
            dist_to_stop = np.clip((current - stop) / current, 0.0, 1.0)

        # Assemble state vector
        state = np.array([
            float(np.clip(price_vs_range, 0.0, 1.0)),       # [0]
            float(np.clip(_momentum(WINDOW_1M),  -0.1, 0.1)),  # [1]
            float(np.clip(_momentum(WINDOW_5M),  -0.2, 0.2)),  # [2]
            float(np.clip(_momentum(WINDOW_15M), -0.3, 0.3)),  # [3]
            float(np.clip(volatility_1h, 0.0, 0.1)),         # [4]
            float(np.clip(volume_norm, -3.0, 3.0)),          # [5]
            float(np.clip(vwap_deviation, -0.05, 0.05)),     # [6]
            in_pos,                                          # [7]
            float(position_pnl),                             # [8]
            float(time_in_pos),                              # [9]
            float(dist_to_stop),                             # [10]
            float(np.clip(trade_frequency, -3.0, 3.0)),      # [11]
            float(np.clip(price_vs_range, 0.0, 1.0)),        # [12]
        ], dtype=np.float32)

        return state
