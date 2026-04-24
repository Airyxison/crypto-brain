"""
Trade Inspector
---------------
Replays the test split deterministically and captures the full context
at every trade entry and exit. Answers the question:

  "Did the agent learn something, or did it get lucky?"

For each trade: feature vector at entry, regime, momentum_8h, price action
before/after, hold duration, exit reason, PnL.

Usage:
  python scripts/trade_inspector.py --db /root/ticks.db --symbol BTCUSDT \
    --checkpoint checkpoints/btcusdt/nova_brain_best.pt
"""

import argparse
import numpy as np
from pathlib import Path

from backtest.runner import load_ticks_from_db, RegimeClassifier
from environment.trading_env import TradingEnv
from environment.order_book import OrderBookSimulator
from features.engineer import FeatureEngineer, MIN_WINDOW
from agent.sac import SAC

FEATURE_NAMES = [
    'price_vs_range',   # [0]
    'momentum_1m',      # [1]
    'momentum_5m',      # [2]
    'momentum_15m',     # [3]
    'volatility_1h',    # [4]
    'volume_norm',      # [5]
    'vwap_deviation',   # [6]
    'in_position',      # [7]
    'position_pnl',     # [8]
    'time_in_position', # [9]
    'dist_to_stop',     # [10]
    'trade_frequency',  # [11]
    'momentum_30m',     # [12]
    'momentum_1d',      # [13]
    'momentum_7d',      # [14]
    'momentum_30d',     # [15]
    'momentum_8h',      # [16]
]

ACTION_NAMES = ['HOLD', 'BUY_LIMIT', 'ADJ_STOP', 'REALIZE_GAIN', 'CANCEL_ORDER']


def inspect(db_path: str, symbol: str, checkpoint: str, context_bars: int = 10):
    # Load data
    ticks = load_ticks_from_db(db_path, symbol=symbol)
    test_ticks = ticks[int(len(ticks) * 0.8):]
    print(f"\n[Inspector] {symbol} | {len(test_ticks):,} test ticks | checkpoint: {checkpoint}")

    # Load agent
    agent = SAC()
    agent.load(checkpoint)

    # Manual replay (mirrors run_backtest deterministic mode)
    eval_config = {'max_episode_steps': len(test_ticks), 'symbol': symbol}
    env = TradingEnv(test_ticks, eval_config)
    env._idx        = MIN_WINDOW
    env._step_count = 0
    env._ob         = OrderBookSimulator(env.initial_cash)
    env._features   = FeatureEngineer()
    for i in range(MIN_WINDOW):
        t = test_ticks[i]
        env._features.update(t['price'], t['volume'], t['trade_time'])
    obs = env._get_obs()

    classifier = RegimeClassifier()
    entry_contexts = {}   # bar_index → context dict captured at BUY_LIMIT
    trade_log = []
    prev_trade_count = 0
    done = False

    while not done:
        idx = env._idx
        price = test_ticks[min(idx, len(test_ticks)-1)]['price']
        regime = classifier.update(price)

        action = agent.select_action(obs, deterministic=True)
        shaped = env._shape_action(action)

        # Capture context at entry
        if shaped == 1:  # BUY_LIMIT
            context_start = max(0, idx - context_bars)
            price_before = [test_ticks[i]['price'] for i in range(context_start, idx)]
            entry_contexts[idx] = {
                'tick_idx':    idx,
                'price':       price,
                'regime':      regime,
                'obs':         obs.copy(),
                'momentum_8h': float(obs[16]),
                'momentum_1d': float(obs[13]),
                'momentum_7d': float(obs[14]),
                'volatility':  float(obs[4]),
                'vwap_dev':    float(obs[6]),
                'price_before': price_before,
                'action_raw':  action,
            }

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Capture completed trades
        current_trades = env._ob.trades
        if len(current_trades) > prev_trade_count:
            for t in current_trades[prev_trade_count:]:
                entry_bar = t['entry_step']
                ctx = entry_contexts.get(entry_bar, {})

                # Price action after entry
                exit_idx = min(entry_bar + t['bars_held'] + context_bars, len(test_ticks)-1)
                price_after = [test_ticks[i]['price'] for i in
                               range(entry_bar + t['bars_held'],
                                     min(entry_bar + t['bars_held'] + context_bars, len(test_ticks)))]

                trade_log.append({**t, **ctx,
                                  'price_after': price_after,
                                  'reward_at_exit': reward})
            prev_trade_count = len(current_trades)

    # Force-close if still open
    if env._ob.position:
        last_price = test_ticks[env._idx - 1]['price']
        env._ob.realize_gain(last_price)
        if len(env._ob.trades) > prev_trade_count:
            t = env._ob.trades[-1]
            entry_bar = t['entry_step']
            ctx = entry_contexts.get(entry_bar, {})
            trade_log.append({**t, **ctx, 'price_after': [], 'reward_at_exit': 0.0})

    # Report
    print(f"\n{'='*70}")
    print(f"  TRADE INSPECTOR — {symbol}")
    print(f"  {len(trade_log)} trades found")
    print(f"{'='*70}")

    if not trade_log:
        print("  No trades. Agent chose to stay in cash the entire test period.")
        return

    for i, t in enumerate(trade_log):
        pnl_pct = t.get('pnl_pct', 0) * 100
        outcome = "WIN ✓" if t.get('pnl', 0) > 0 else "LOSS ✗"
        print(f"\n  ── Trade {i+1} of {len(trade_log)} ─── {outcome} ──────────────────")
        print(f"  Entry bar:    {t.get('tick_idx', t.get('entry_step', '?')):,}")
        print(f"  Entry price:  ${t.get('entry_price', 0):,.4f}")
        print(f"  Exit price:   ${t.get('exit_price', 0):,.4f}")
        print(f"  PnL:          {pnl_pct:+.4f}%  (${t.get('pnl', 0):+.4f})")
        print(f"  Hold:         {t.get('bars_held', 0)} bars")
        print(f"  Exit reason:  {t.get('reason', '?')}")
        print(f"  Regime:       {t.get('regime', '?')}")
        print(f"\n  ── Feature Vector at Entry ──")
        obs = t.get('obs')
        if obs is not None:
            key_features = [0, 1, 2, 3, 4, 6, 12, 13, 14, 15, 16]
            for fi in key_features:
                bar = '█' * int(abs(obs[fi]) * 20) if abs(obs[fi]) <= 1 else '█' * 20
                sign = '+' if obs[fi] >= 0 else '-'
                print(f"    [{fi:2d}] {FEATURE_NAMES[fi]:<20} {sign}{abs(obs[fi]):.5f}  {bar}")

        # Price action context
        before = t.get('price_before', [])
        after  = t.get('price_after', [])
        if before:
            move_before = (before[-1] - before[0]) / (before[0] + 1e-9) * 100
            print(f"\n  ── Price Context ──")
            print(f"  {context_bars} bars before entry: {move_before:+.3f}% move  "
                  f"(${before[0]:,.2f} → ${before[-1]:,.2f})")
        if after:
            move_after = (after[-1] - after[0]) / (after[0] + 1e-9) * 100
            print(f"  {len(after)} bars after exit:  {move_after:+.3f}% move  "
                  f"(${after[0]:,.2f} → ${after[-1]:,.2f})")

    # Cross-trade pattern summary
    print(f"\n{'='*70}")
    print(f"  PATTERN SUMMARY")
    print(f"{'='*70}")
    wins  = [t for t in trade_log if t.get('pnl', 0) > 0]
    loss  = [t for t in trade_log if t.get('pnl', 0) <= 0]
    print(f"  Win/Loss:     {len(wins)}W / {len(loss)}L")

    if trade_log and trade_log[0].get('obs') is not None:
        for fi in [16, 13, 14, 4]:  # momentum_8h, 1d, 7d, volatility
            vals = [t['obs'][fi] for t in trade_log if t.get('obs') is not None]
            if vals:
                print(f"  {FEATURE_NAMES[fi]:<20} at entry: "
                      f"mean={np.mean(vals):+.5f}  "
                      f"min={np.min(vals):+.5f}  "
                      f"max={np.max(vals):+.5f}")

    regimes = [t.get('regime', 'UNKNOWN') for t in trade_log]
    from collections import Counter
    print(f"  Regimes at entry: {dict(Counter(regimes))}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--db',         default='/root/ticks.db')
    p.add_argument('--symbol',     default='BTCUSDT')
    p.add_argument('--checkpoint', default='checkpoints/btcusdt/nova_brain_best.pt')
    p.add_argument('--context',    type=int, default=10,
                   help='Bars of price context before/after each trade')
    args = p.parse_args()
    inspect(args.db, args.symbol, args.checkpoint, args.context)
