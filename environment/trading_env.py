"""
Trading Environment
-------------------
Gymnasium-compatible environment wrapping the order book simulator
and feature engineer. Steps through historical tick data.

Observation: 13-dim float32 vector (see features/engineer.py)
Action:      Discrete(5)
  0 = HOLD
  1 = PLACE_BUY_LIMIT
  2 = ADJUST_STOP
  3 = REALIZE_GAIN
  4 = CANCEL_ORDER

Reward: Sortino-influenced, asymmetric downside punishment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from features.engineer import FeatureEngineer, MIN_WINDOW
from environment.order_book import OrderBookSimulator


# Action constants
HOLD          = 0
BUY_LIMIT     = 1
ADJUST_STOP   = 2
REALIZE_GAIN  = 3
CANCEL_ORDER  = 4

# Reward hyperparameters
ALPHA = 0.5    # drawdown penalty weight
BETA  = 0.10   # stop-loss hit penalty
GAMMA = 0.0001 # losing hold cost per bar


class TradingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, ticks: list[dict], config: dict | None = None):
        """
        ticks: list of dicts with keys: price (float), volume (float), trade_time (int ms)
        config: optional overrides for initial_cash, max_hold_bars
        """
        super().__init__()
        self.ticks  = ticks
        self.config = config or {}

        self.initial_cash  = self.config.get('initial_cash', 10_000.0)
        self.max_hold_bars = self.config.get('max_hold_bars', 1440)  # ~24h of ticks

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._idx      = 0
        self._ob       = OrderBookSimulator(self.initial_cash)
        self._features = FeatureEngineer()

    # -------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a random point deep enough to have full feature history
        max_start = max(MIN_WINDOW, len(self.ticks) - self.max_hold_bars - 1)
        start = self.np_random.integers(MIN_WINDOW, max_start) if max_start > MIN_WINDOW else MIN_WINDOW

        self._idx      = start
        self._ob       = OrderBookSimulator(self.initial_cash)
        self._features = FeatureEngineer()

        # Pre-warm feature engineer with history before start
        for i in range(max(0, start - MIN_WINDOW), start):
            t = self.ticks[i]
            self._features.update(t['price'], t['volume'], t['trade_time'])

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        tick = self.ticks[self._idx]
        price  = tick['price']
        volume = tick['volume']
        t_ms   = tick['trade_time']

        prev_pv = self._ob.portfolio_value

        # Execute action
        self._execute_action(action, price)

        # Advance tick
        event = self._ob.tick(price)
        self._features.update(price, volume, t_ms)
        self._idx += 1

        # Compute reward
        reward = self._compute_reward(prev_pv, event)

        # Check termination
        terminated = (
            self._idx >= len(self.ticks) - 1
            or self._ob.cash <= 0
        )
        truncated = False

        obs  = self._get_obs()
        info = {
            'portfolio_value': self._ob.portfolio_value,
            'realized_pnl':    self._ob.realized_pnl,
            'in_position':     self._ob.position is not None,
            'stop_triggered':  event.get('stop_hit', False),
        }

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------

    def _execute_action(self, action: int, price: float):
        if action == BUY_LIMIT:
            self._ob.place_buy_limit(price)
        elif action == ADJUST_STOP:
            self._ob.adjust_stop(price)
        elif action == REALIZE_GAIN:
            self._ob.realize_gain(price)
        elif action == CANCEL_ORDER:
            self._ob.cancel_order()
        # HOLD: no-op

    def _compute_reward(self, prev_pv: float, event: dict) -> float:
        current_pv = self._ob.portfolio_value

        # Base return (signed, fractional)
        base = (current_pv - prev_pv) / (prev_pv + 1e-9)

        # Drawdown penalty — asymmetric, 2% free zone
        drawdown = (self._ob.portfolio_peak - current_pv) / (self._ob.portfolio_peak + 1e-9)
        drawdown_penalty = ALPHA * max(0.0, drawdown - 0.02)

        # Stop-loss hit — capital preservation is a hard constraint
        stop_penalty = -BETA if event.get('stop_hit') else 0.0

        # Small cost for holding a losing position (encourages decisive exits)
        hold_cost = 0.0
        if self._ob.position:
            pnl_pct = self._ob.position.unrealized_pnl_pct(self.ticks[self._idx - 1]['price'])
            if pnl_pct < -0.01:
                hold_cost = -GAMMA

        return float(base - drawdown_penalty + stop_penalty + hold_cost)

    def _get_obs(self) -> np.ndarray:
        pos = self._ob.position
        position_state = {
            'in_position': pos is not None,
            'entry_price': pos.entry_price if pos else None,
            'stop_price':  pos.stop_price  if pos else None,
            'bars_held':   (self._idx - pos.entry_bar) if pos else 0,
            'max_hold':    self.max_hold_bars,
        }
        if self._features.ready:
            return self._features.extract(position_state)
        return np.zeros(13, dtype=np.float32)
