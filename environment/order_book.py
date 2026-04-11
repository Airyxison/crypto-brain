"""
Order Book Simulator
--------------------
Simulates realistic limit order behavior during backtesting.

Rules:
  - Limit BUY fills when market price drops to or below the limit price
  - Limit SELL fills when market price rises to or above the limit price
  - Orders expire after `expiry_bars` ticks if unfilled
  - Slippage: fills at the limit price (conservative — no partial fills in POC)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LimitOrder:
    side: str           # 'buy' or 'sell'
    limit_price: float
    size: float         # fraction of capital (0-1)
    placed_at: int      # bar index when placed
    expiry_bars: int = 120  # expire after 2 minutes of ticks by default

    def is_expired(self, current_bar: int) -> bool:
        return (current_bar - self.placed_at) >= self.expiry_bars

    def would_fill(self, current_price: float) -> bool:
        if self.side == 'buy':
            return current_price <= self.limit_price
        else:
            return current_price >= self.limit_price


@dataclass
class Position:
    entry_price: float
    size: float         # units held
    cost_basis: float   # total cash spent
    entry_bar: int
    stop_price: float   # dynamic stop-loss level
    peak_price: float   # highest price seen since entry (for trailing stop)

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def update_peak(self, current_price: float):
        if current_price > self.peak_price:
            self.peak_price = current_price

    def is_stop_triggered(self, current_price: float) -> bool:
        return current_price <= self.stop_price


class OrderBookSimulator:
    """
    Stateful order book for one asset.
    Call tick() on each new price to process fills and stops.
    """

    # Default stop: 2% below entry.
    # Limit disc = 0 for POC: fills on next tick (market-like, eliminates credit delay)
    DEFAULT_STOP_PCT    = 0.005
    DEFAULT_LIMIT_DISC  = 0.0
    FEE_RATE            = 0.001  # 0.1% per side (Coinbase taker fee)

    def __init__(self, initial_cash: float = 10_000.0):
        self.cash: float              = initial_cash
        self.position: Optional[Position] = None
        self.pending_order: Optional[LimitOrder] = None
        self.current_bar: int         = 0

        # Metrics
        self.realized_pnl: float      = 0.0
        self.portfolio_peak: float    = initial_cash
        self.stop_triggered: bool     = False
        self.trades: list             = []

    @property
    def portfolio_value(self) -> float:
        if self.position:
            return self.cash + self.position.size * self._last_price
        return self.cash

    def tick(self, price: float) -> dict:
        """
        Process one tick. Returns an event dict describing what happened.
        """
        self._last_price = price
        self.current_bar += 1
        self.stop_triggered = False
        event = {'filled': False, 'stop_hit': False, 'expired': False}

        # 1. Check pending order fill
        if self.pending_order and not self.position:
            if self.pending_order.would_fill(price):
                self._fill_buy(price)
                event['filled'] = True
            elif self.pending_order.is_expired(self.current_bar):
                self.pending_order = None
                event['expired'] = True

        # 2. Check stop-loss on open position
        if self.position:
            self.position.update_peak(price)
            if self.position.is_stop_triggered(price):
                self._trigger_stop(price)
                event['stop_hit'] = True

        # 3. Update portfolio peak
        pv = self.portfolio_value
        if pv > self.portfolio_peak:
            self.portfolio_peak = pv

        return event

    # -------------------------------------------------------------------------
    # Actions (called by the RL agent via the environment)
    # -------------------------------------------------------------------------

    def place_buy_limit(self, current_price: float, capital_fraction: float = 0.9) -> bool:
        """Place a limit buy order below current price. Returns True if placed."""
        if self.pending_order or self.position:
            return False  # already have order or position
        if self.cash <= 0:
            return False

        limit_price = current_price * (1.0 - self.DEFAULT_LIMIT_DISC)
        size = (self.cash * capital_fraction) / limit_price

        self.pending_order = LimitOrder(
            side='buy',
            limit_price=limit_price,
            size=size,
            placed_at=self.current_bar,
        )
        return True

    def adjust_stop(self, current_price: float) -> bool:
        """Trail stop up to lock in gains. Only tightens, never loosens."""
        if not self.position:
            return False

        # Trail: stop = max(current_stop, peak_price * (1 - stop_pct))
        new_stop = self.position.peak_price * (1.0 - self.DEFAULT_STOP_PCT)
        if new_stop > self.position.stop_price:
            self.position.stop_price = new_stop
        return True

    def realize_gain(self, current_price: float) -> bool:
        """Close position at market (fills at current price, no limit)."""
        if not self.position:
            return False
        self._close_position(current_price, reason='realized')
        return True

    def cancel_order(self) -> bool:
        """Cancel any pending unfilled limit order."""
        if not self.pending_order:
            return False
        self.pending_order = None
        return True

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _fill_buy(self, fill_price: float):
        cost = self.pending_order.size * fill_price
        entry_fee = cost * self.FEE_RATE
        total_cost = cost + entry_fee
        self.cash -= total_cost
        self.position = Position(
            entry_price=fill_price,
            size=self.pending_order.size,
            cost_basis=total_cost,  # fee baked in so PnL is after-fee
            entry_bar=self.current_bar,
            stop_price=fill_price * (1.0 - self.DEFAULT_STOP_PCT),
            peak_price=fill_price,
        )
        self.pending_order = None

    def _trigger_stop(self, price: float):
        self.stop_triggered = True
        self._close_position(price, reason='stop_loss')

    def _close_position(self, price: float, reason: str):
        if not self.position:
            return
        proceeds = self.position.size * price
        exit_fee = proceeds * self.FEE_RATE
        net_proceeds = proceeds - exit_fee
        pnl = net_proceeds - self.position.cost_basis
        self.realized_pnl += pnl
        self.cash += net_proceeds
        self.trades.append({
            'entry_price': self.position.entry_price,
            'exit_price': price,
            'size': self.position.size,
            'pnl': pnl,
            'pnl_pct': pnl / self.position.cost_basis,
            'bars_held': self.current_bar - self.position.entry_bar,
            'reason': reason,
        })
        self.position = None
