"""
Trading Environment
-------------------
Gymnasium-compatible environment wrapping the order book simulator
and feature engineer. Steps through historical tick data.

Observation: 16-dim float32 vector (see features/engineer.py)
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
ALPHA               = 0.5    # drawdown penalty weight
BETA                = 0.0    # stop-loss hit penalty (removed: base return already captures the loss)
GAMMA               = 0.0001 # losing hold cost per bar
EPSILON             = 0.0001 # opportunity cost: penalty for holding cash while market moves
MIN_HOLD_BARS       = 50     # minimum bars before a realized-gain bonus is awarded (was 20 — discourage scalping)
INVALID_ACTION_COST = 0.0001 # penalty for no-op actions (REALIZE/CANCEL/ADJ_STOP with nothing to act on)


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

        self.initial_cash      = self.config.get('initial_cash', 10_000.0)
        self.max_hold_bars     = self.config.get('max_hold_bars', 300)
        self.max_episode_steps = self.config.get('max_episode_steps', 500)  # longer episodes let the agent see trades through
        self._step_count       = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
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

        self._idx        = start
        self._step_count = 0
        self._ob         = OrderBookSimulator(self.initial_cash)
        self._features   = FeatureEngineer()

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

        prev_pv           = self._ob.portfolio_value
        prev_realized_pnl = self._ob.realized_pnl  # track for realized-gain bonus
        # Bars held before this action fires (used by _compute_reward for min-hold check)
        prev_bars_held    = (self._idx - self._ob.position.entry_bar) if self._ob.position else 0

        # Execute action — returns True if the action had a valid target
        action_valid = self._execute_action(action, price)

        # Advance tick
        event = self._ob.tick(price)
        self._features.update(price, volume, t_ms)
        self._idx       += 1
        self._step_count += 1

        # Realized PnL delta this step — positive only when a trade just closed
        step_realized = self._ob.realized_pnl - prev_realized_pnl

        # Compute reward — scaled up so Q-network can differentiate
        reward = self._compute_reward(prev_pv, event, step_realized, prev_bars_held, action_valid) * 1000.0

        # Check termination
        terminated = (
            self._idx >= len(self.ticks) - 1
            or self._ob.cash <= 0
        )
        # Truncate episode after max_episode_steps to get many short episodes
        truncated = self._step_count >= self.max_episode_steps

        obs  = self._get_obs()
        info = {
            'portfolio_value': self._ob.portfolio_value,
            'realized_pnl':    self._ob.realized_pnl,
            'in_position':     self._ob.position is not None,
            'stop_triggered':  event.get('stop_hit', False),
        }

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------

    def _execute_action(self, action: int, price: float) -> bool:
        """Execute action. Returns True if the action had a valid target, False if it was a no-op."""
        if action == BUY_LIMIT:
            vol = self._features.current_volatility if self._features.ready else 0.0
            return self._ob.place_buy_limit(price, volatility=vol)
        elif action == ADJUST_STOP:
            return self._ob.adjust_stop(price)
        elif action == REALIZE_GAIN:
            return self._ob.realize_gain(price)
        elif action == CANCEL_ORDER:
            return self._ob.cancel_order()
        # HOLD: always valid
        return True

    def _compute_reward(self, prev_pv: float, event: dict, step_realized: float = 0.0, bars_held: int = 0, action_valid: bool = True) -> float:
        current_pv = self._ob.portfolio_value

        # Base return (signed, fractional)
        base = (current_pv - prev_pv) / (prev_pv + 1e-9)

        # Drawdown penalty — asymmetric, 2% free zone
        drawdown = (self._ob.portfolio_peak - current_pv) / (self._ob.portfolio_peak + 1e-9)
        drawdown_penalty = ALPHA * max(0.0, drawdown - 0.02)

        # Stop-loss hit — capital preservation is a hard constraint
        stop_penalty = -BETA if event.get('stop_hit') else 0.0

        # Cost for holding a losing position (encourages decisive exits)
        hold_cost = 0.0
        if self._ob.position:
            pnl_pct = self._ob.position.unrealized_pnl_pct(self.ticks[self._idx - 1]['price'])
            if pnl_pct < -0.01:
                hold_cost = -GAMMA

        # Opportunity cost: penalize sitting in cash when price trends up.
        # NOTE: no inner *1000 — the outer *1000 in step() is the only amplifier.
        # Previous version had *1e6 compound scale which overwhelmed the base signal.
        opp_cost = 0.0
        if not self._ob.position and not self._ob.pending_order and self._idx >= 2:
            prev_price = self.ticks[self._idx - 2]['price']
            curr_price = self.ticks[self._idx - 1]['price']
            price_move = (curr_price - prev_price) / (prev_price + 1e-9)
            if price_move > 0:
                opp_cost = -EPSILON * price_move

        # Realized gain bonus — explicit credit assignment when a trade closes.
        # Requires MIN_HOLD_BARS held before awarding any bonus; premature exits
        # get a small penalty to discourage scalping sub-penny moves after fees.
        realized_bonus = 0.0
        if step_realized != 0.0:
            pct = step_realized / (prev_pv + 1e-9)
            if bars_held >= MIN_HOLD_BARS:
                realized_bonus = pct  # 1:1, no amplifier — fees do the heavy lifting
            else:
                realized_bonus = -abs(pct) * 0.5  # premature exit penalty

        # Penalty for no-op actions — discourages calling REALIZE/CANCEL/ADJUST with nothing to act on
        invalid_penalty = 0.0 if action_valid else -INVALID_ACTION_COST

        return float(base - drawdown_penalty + stop_penalty + hold_cost + opp_cost + realized_bonus + invalid_penalty)

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
        return np.zeros(16, dtype=np.float32)
