# PRD-004 — Live Execution Integration with crypto-engine

**Status:** Future — requires PRD-001, PRD-002, and PRD-003 first  
**Priority:** P2 — Final milestone before real capital deployment  
**Author:** Nova  
**Date:** 2026-04-10

---

## Problem

The crypto-brain agent currently exists entirely in simulation. It trains and backtests against historical tick data in SQLite, but has no path to actually place orders. The crypto-engine (Rust) collects live tick data and writes it to `ticks.db`, but the brain does not read from it in real time and has no mechanism to send orders back to an exchange.

This PRD defines the integration architecture needed to go from "backtest-validated policy" to "live trading system."

---

## Goal

Connect the trained SAC agent to the live crypto-engine tick feed and a real (or paper) exchange order API, so that agent decisions result in actual order placement and management.

**Scope of this PRD:** Architecture definition, interface contracts, and the minimum viable integration path. Full exchange connectivity and capital deployment are gated on PRD-002's drawdown target being met.

---

## Prerequisites

1. PRD-001 (best_sortino bug fix) — ensures the production checkpoint is reliable.
2. PRD-002 (drawdown reduction) — max drawdown < -15% before any real capital is at risk.
3. The production checkpoint must pass a forward-walk validation (see below) — not just a standard backtest on the held-out 20%.
4. A paper trading account or testnet must be set up and validated before real capital.

---

## System Architecture

### Current State

```
crypto-engine (Rust)
    → writes ticks → ticks.db (SQLite)

crypto-brain (Python)
    reads ticks.db → Feature Engineer → TradingEnv → SAC → simulated orders
```

### Target State

```
Exchange WebSocket
    ↓  live ticks
crypto-engine (Rust)
    ↓  writes ticks  →  ticks.db (SQLite)
    ↓  streams new ticks via IPC (Unix socket / named pipe / Redis)
    ↓
crypto-brain inference loop (Python)
    ↓  Feature Engineer (stateful, warm from history)
    ↓  SAC.select_action(obs, deterministic=True)
    ↓  action → Order Manager
    ↓
Order Manager (Python)
    ↓  REST API calls
    ↓
Exchange (paper or real)
    ↓  order confirmations
    ↓
Position Tracker (Python)
    ↓  updates position state → feeds back into Feature Engineer
```

---

## Component Design

### 1. Tick Streaming (crypto-engine side)

The Rust engine currently writes ticks to SQLite. It needs to additionally emit ticks in real time to the Python brain.

**Recommended approach:** Unix domain socket or named pipe. The engine writes a newline-delimited JSON stream:

```json
{"symbol":"BTCUSDT","price":84231.5,"volume":0.03,"trade_time":1744320000000}
```

**Alternative:** Poll `ticks.db` on a short interval (500ms). Simpler but adds latency and SQL overhead. Acceptable for POC, not production.

**Rust changes required:**
- Add a second write path alongside the existing SQLite insert: serialize tick to JSON, write to socket/pipe.
- Use a background thread to avoid blocking the main tick loop.

### 2. Inference Loop (crypto-brain side)

New file: `live/inference_loop.py`

Responsibilities:
- Connect to the tick stream socket.
- Maintain a warm `FeatureEngineer` (pre-seeded from recent `ticks.db` history on startup).
- On each tick: call `features.update()`, extract observation, call `agent.select_action(obs, deterministic=True)`.
- Pass action to `OrderManager`.
- Receive position state updates from `OrderManager` and inject into feature extraction.

Key design constraint: the inference loop must be single-threaded and synchronous with the tick stream. The SAC inference is fast (< 1ms on CPU), so this is not a bottleneck.

```python
# Pseudocode
agent = SAC()
agent.load('checkpoints/nova_brain_btc_best.pt')

features = FeatureEngineer()
seed_features_from_db(features, db_path, symbol, lookback=WINDOW_4H)

position_state = {'in_position': False, 'entry_price': None, ...}

for tick in stream_ticks(socket_path):
    features.update(tick['price'], tick['volume'], tick['trade_time'])
    if not features.ready:
        continue
    obs = features.extract(position_state)
    action = agent.select_action(obs, deterministic=True)
    order_manager.handle(action, tick['price'])
    position_state = order_manager.position_state
```

### 3. Order Manager

New file: `live/order_manager.py`

Wraps exchange API calls. Mirrors the logic of `OrderBookSimulator` but calls real REST endpoints instead of simulating fills.

Key methods (matching `OrderBookSimulator` interface):
- `place_buy_limit(price)` → POST to exchange, return order_id
- `adjust_stop(price)` → modify or replace stop order
- `realize_gain(price)` → market sell
- `cancel_order()` → cancel pending order

The `OrderManager` must be exchange-agnostic at the interface level, with concrete implementations for:
- `CoinbaseOrderManager` (Coinbase Advanced Trade API)
- `PaperOrderManager` (local simulation, no real API calls — for testing)

`PaperOrderManager` should produce identical behavior to `OrderBookSimulator` so that backtests and paper trades can be directly compared.

### 4. Position Tracker

The `OrderManager` maintains authoritative position state and exposes it as a dict matching the format expected by `FeatureEngineer.extract()`:

```python
{
    'in_position': True,
    'entry_price': 84100.0,
    'stop_price':  83259.0,
    'bars_held':   14,
    'max_hold':    300,
}
```

This closes the feedback loop: live position state feeds directly back into the observation vector with no simulation layer.

---

## Safety Requirements

### Hard Limits (enforced in OrderManager, not configurable at runtime)

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max position size | 90% of cash | Same as simulation |
| Max open orders | 1 | Agent was trained single-position |
| Max daily loss | -5% of starting capital | Kill switch before catastrophic loss |
| Max drawdown (live) | -15% | Matches PRD-002 target |
| Emergency halt | Manual kill switch | Human override always available |

If any hard limit is breached, the `OrderManager` must:
1. Close any open position at market.
2. Cancel any pending orders.
3. Write a `HALT` event to a log file.
4. Stop accepting new actions until manually reset.

### Separate Paper Phase

Before trading real capital:
1. Run inference loop against paper trading account for at least 30 days.
2. Paper trading results must show Sortino > 1.5 and drawdown < -15% over that window.
3. Eric reviews and explicitly approves live capital deployment.

This paper phase is non-negotiable. Backtests are in-sample even with train/test splits — live performance is the only honest validation.

---

## Forward-Walk Validation

Before live deployment, run a forward walk on the most recent 10% of available tick data (data the model was never trained or tested on):

```python
# Reserve the last 10% of ticks — separate from the 20% test split
forward_ticks = all_ticks[int(len(all_ticks) * 0.9):]
results = run_backtest(agent, forward_ticks, verbose=True)
```

Acceptance criteria:
- Forward-walk Sortino > 1.0 (lower bar than backtest — forward data is genuinely unseen)
- Forward-walk max drawdown < -20%
- Forward-walk total return > 0% (positive)

If forward-walk fails, do not proceed to live execution regardless of backtest metrics.

---

## Monitoring and Observability

The live system must emit structured logs that the `crypto-ui` dashboard can consume:

```json
{
  "timestamp": 1744320000000,
  "event": "order_placed",
  "symbol": "BTCUSDT",
  "action": "BUY_LIMIT",
  "price": 84100.0,
  "size": 0.0107,
  "portfolio_value": 10241.50
}
```

Event types: `tick_received`, `order_placed`, `order_filled`, `order_expired`, `stop_triggered`, `position_closed`, `daily_pnl`, `halt`.

Dashboard requirements (separate from this PRD, but noted for coordination with `crypto-ui`):
- Real-time portfolio value chart
- Open position display (entry, current price, stop, unrealized P&L)
- Trade history table
- Action distribution histogram
- Halt status indicator

---

## Files to Create

| File | Description |
|------|-------------|
| `live/__init__.py` | Package init |
| `live/inference_loop.py` | Main live inference loop |
| `live/order_manager.py` | Exchange-agnostic order interface + PaperOrderManager |
| `live/coinbase_order_manager.py` | Coinbase Advanced Trade implementation |
| `live/monitor.py` | Structured event logger |
| `live/config.py` | Live config (API keys via env vars, limits) |

---

## Files to Modify

| File | Change |
|------|--------|
| `agent/sac.py` | Confirm `select_action(deterministic=True)` works in live context (no changes expected) |
| `features/engineer.py` | Add `seed_from_db()` helper for warm startup |
| `environment/order_book.py` | Potentially extract interface to share with `PaperOrderManager` |

---

## Acceptance Criteria

1. `live/inference_loop.py` runs against `PaperOrderManager` for 30 days without errors or unexpected halts.
2. Paper trading metrics over 30 days: Sortino > 1.0, max drawdown < -20%, positive return.
3. Forward-walk backtest on most recent 10% of ticks passes (Sortino > 1.0, drawdown < -20%, return > 0%).
4. Hard limits are enforced: daily loss halt tested explicitly with a simulated losing streak.
5. Eric reviews paper results and approves live capital deployment.
6. `crypto-ui` dashboard displays live position and trade history in real time.

---

## Notes

- API keys must never be committed to the repository. Use environment variables only.
- The `PaperOrderManager` must be the default in all code paths. Switching to `CoinbaseOrderManager` requires an explicit flag (`--live` or env var `NOVA_LIVE=1`).
- Start with a small capital allocation (e.g., $500) for the first live month regardless of paper performance. Increase only after validating live behavior matches paper.
- This PRD covers single-asset (BTC) live execution. Multi-asset live trading (PRD-003 extension) is a separate future milestone.

## Future Signal Enhancements (Post-Launch)

**Orderbook data** is worth revisiting once the live system is running. At inference time, a real-time orderbook snapshot (bid/ask imbalance, depth within 0.5% of mid, large resting orders) could add meaningful signal — particularly for predicting short-term reversals and avoiding entries into spoofed walls. This signal is most valuable at live inference resolution, not at 1-min candle aggregation.

Not a priority now because: (1) historical orderbook data can't be backfilled, so it can't improve training until months of collection have accumulated; (2) crypto spoofing degrades the signal significantly and the model would need to learn to discount it; (3) the current bottleneck is drawdown, not signal quality.

Revisit after 30-day paper phase is complete and live execution is stable.
