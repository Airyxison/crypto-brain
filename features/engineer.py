"""
Feature Engineering
-------------------
Converts raw tick data into a normalized 19-dimensional state vector.
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
  [12] momentum_30m        — 30-min return (fills gap between 15m and 1d)
  [13] momentum_1d         — 24h return (macro intraday trend)
  [14] momentum_7d         — 7-day return (weekly regime)
  [15] momentum_30d        — 30-day return (macro bull/bear regime)
  [16] momentum_8h         — 8-hour return (explicit regime signal, matches REGIME_WINDOW=480)
  [17] vol_flow_30         — 30-bar directional volume flow [-1, +1] (Barney v1, 2026-04-25)
  [18] vol_flow_240        — 240-bar directional volume flow, lagged t-240 to t-31 [-1, +1]
                             (background regime — structurally decorrelated from vol_flow_30)
"""

import numpy as np
from collections import deque


# Tick window requirements (all in bars = 1-minute candles)
WINDOW_1M   = 60
WINDOW_5M   = 300
WINDOW_15M  = 900
WINDOW_1H   = 3_600
WINDOW_4H   = 14_400
WINDOW_1D   = 1_440    # 24 hours
WINDOW_7D   = 10_080   # 7 days
WINDOW_30D  = 43_200   # 30 days — macro regime window
WINDOW_8H   = 480      # 8-hour window — matches REGIME_WINDOW in trading_env.py
MIN_WINDOW  = WINDOW_5M   # 5 min of history to start — regime features degrade gracefully

# Volume flow windows (Barney v1, 2026-04-25)
VOL_FLOW_SHORT  = 30   # 30-bar immediate pressure window
VOL_FLOW_LONG   = 240  # 240-bar background regime (lagged: t-240 to t-31, decorrelated from short)
VOL_FLOOR_WINDOW = 1_440  # 24h median for thin-tape floor computation


class FeatureEngineer:
    def __init__(self):
        self.prices     = deque(maxlen=WINDOW_30D)  # 30 days — supports macro regime features
        self.volumes    = deque(maxlen=WINDOW_30D)
        self.times      = deque(maxlen=WINDOW_30D)
        self._signed_vols = deque(maxlen=WINDOW_30D)  # bar_sign * quantity per bar
        self._ready     = False

    def update(self, price: float, volume: float, trade_time: int):
        # bar_sign: direction of this bar vs previous (Bug 1 fix: fillna(0) on first bar)
        if self.prices:
            bar_sign = float(np.sign(price - self.prices[-1]))
        else:
            bar_sign = 0.0
        self.prices.append(price)
        self.volumes.append(volume)
        self.times.append(trade_time)
        self._signed_vols.append(bar_sign * abs(volume))
        if len(self.prices) >= MIN_WINDOW:
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def _vol_flow(self, window: int, lag_end: int = 0) -> float:
        """
        Directional volume flow over `window` bars ending `lag_end` bars ago.
        Returns value in [-1, +1]. Returns 0.0 if insufficient history.

        Barney v1 implementation — four bugs fixed:
          1. bar_sign fillna(0) handled in update()
          2. epsilon guard on denominator (Bug 2)
          3. min_periods enforced via explicit length check (Bug 4)
          4. thin-tape volume floor — 20% of 24h median * window (design conflict fix)
        Bug 3 (cross-symbol leakage) is not applicable here — FeatureEngineer
        is always single-symbol (one instance per env).
        """
        n = len(self._signed_vols)
        required = window + lag_end
        if n < required:
            return 0.0

        sv   = np.array(self._signed_vols, dtype=np.float64)
        vols = np.array(self.volumes,      dtype=np.float64)

        # Slice: lag_end=0 → last `window` bars; lag_end=31 → t-240 to t-31
        if lag_end > 0:
            sv_window   = sv[-(required): -(lag_end)]
            vol_window  = vols[-(required): -(lag_end)]
        else:
            sv_window   = sv[-window:]
            vol_window  = vols[-window:]

        flow_sum = float(sv_window.sum())
        vol_sum  = float(vol_window.sum())

        # Thin-tape volume floor: 20% of 24h median × window (Bug: design conflict fix)
        if n >= VOL_FLOOR_WINDOW:
            median_vol = float(np.median(vols[-VOL_FLOOR_WINDOW:]))
        else:
            median_vol = float(np.median(vols))
        vol_floor = median_vol * 0.2 * window
        vol_sum   = max(vol_sum, vol_floor) + 1e-8  # Bug 2: epsilon guard

        return float(np.clip(flow_sum / vol_sum, -1.0, 1.0))

    @property
    def momentum_8h(self) -> float:
        """8-hour rolling return. Diagnostic use only — do NOT call from _compute_reward
        (one step stale due to call order: _compute_reward runs before extract)."""
        prices = self.prices
        if len(prices) >= 480:
            base = prices[-480]
            return (prices[-1] - base) / (base + 1e-9)
        return 0.0

    @property
    def current_volatility(self) -> float:
        """Raw (unclipped) 1h realized volatility — for stop-loss sizing at entry.
        Returns 0.0 if not enough history yet."""
        if not self._ready or len(self.prices) < 2:
            return 0.0
        prices = np.array(self.prices, dtype=np.float64)
        n1h = min(WINDOW_1H, len(prices))
        returns_1h = np.diff(prices[-n1h:]) / prices[-n1h:-1]
        return float(returns_1h.std()) if len(returns_1h) > 1 else 0.0

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

        # Macro regime features — multi-timescale momentum so the model can
        # distinguish short noise from a week-long trend from a 30-day bear market.
        # Gracefully degrades: if less than N bars of history exist, uses what's available.
        momentum_1d  = _momentum(WINDOW_1D)
        momentum_7d  = _momentum(WINDOW_7D)
        momentum_30d = _momentum(WINDOW_30D)
        momentum_8h  = _momentum(WINDOW_8H)

        # Volume flow features (Barney v1, 2026-04-25)
        vol_flow_30  = self._vol_flow(VOL_FLOW_SHORT, lag_end=0)   # immediate pressure
        vol_flow_240 = self._vol_flow(VOL_FLOW_LONG,  lag_end=31)  # background regime (lagged)

        # Assemble state vector
        state = np.array([
            float(np.clip(price_vs_range, 0.0, 1.0)),          # [0]
            float(np.clip(_momentum(WINDOW_1M),  -0.1, 0.1)),  # [1]
            float(np.clip(_momentum(WINDOW_5M),  -0.2, 0.2)),  # [2]
            float(np.clip(_momentum(WINDOW_15M), -0.3, 0.3)),  # [3]
            float(np.clip(volatility_1h, 0.0, 0.1)),           # [4]
            float(np.clip(volume_norm, -3.0, 3.0)),            # [5]
            float(np.clip(vwap_deviation, -0.05, 0.05)),       # [6]
            in_pos,                                            # [7]
            float(position_pnl),                               # [8]
            float(time_in_pos),                                # [9]
            float(dist_to_stop),                               # [10]
            float(np.clip(trade_frequency, -3.0, 3.0)),        # [11]
            float(np.clip(_momentum(1800), -0.25, 0.25)),      # [12] 30-min momentum
            float(np.clip(momentum_1d,  -0.2, 0.2)),           # [13] 24h trend
            float(np.clip(momentum_7d,  -0.5, 0.5)),           # [14] weekly regime
            float(np.clip(momentum_30d, -1.0, 1.0)),           # [15] macro bull/bear
            float(np.clip(momentum_8h,  -0.5, 0.5)),           # [16] 8h regime signal
            float(vol_flow_30),                                # [17] 30-bar vol flow
            float(vol_flow_240),                               # [18] 240-bar vol flow (lagged)
        ], dtype=np.float32)

        return state
