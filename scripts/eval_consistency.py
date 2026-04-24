"""
Evaluation Consistency Test
----------------------------
Runs the same checkpoint N times against random windows of the test split.
Answers: "Is the agent's behavior consistent, or luck-dependent on episode start?"

If trade patterns repeat across seeds — learned behavior.
If results scatter randomly — lucky on specific market conditions.

Usage:
  python scripts/eval_consistency.py --db /root/ticks.db --symbol BTCUSDT \
    --checkpoint checkpoints/btcusdt/nova_brain_best.pt --runs 10 --window 10000
"""

import argparse
import numpy as np
from collections import defaultdict

from backtest.runner import load_ticks_from_db, compute_sortino, compute_max_drawdown, RegimeClassifier
from environment.trading_env import TradingEnv
from environment.order_book import OrderBookSimulator
from features.engineer import FeatureEngineer, MIN_WINDOW
from agent.sac import SAC


def run_window(agent, ticks, start_idx, window_size, symbol='BTCUSDT'):
    """Run one evaluation window deterministically from start_idx."""
    window = ticks[start_idx: start_idx + window_size]
    if len(window) < MIN_WINDOW + 100:
        return None

    eval_config = {'max_episode_steps': len(window), 'symbol': symbol}
    env = TradingEnv(window, eval_config)
    env._idx        = MIN_WINDOW
    env._step_count = 0
    env._ob         = OrderBookSimulator(env.initial_cash)
    env._features   = FeatureEngineer()
    for i in range(MIN_WINDOW):
        t = window[i]
        env._features.update(t['price'], t['volume'], t['trade_time'])
    obs = env._get_obs()

    portfolio_values = [env._ob.portfolio_value]
    step_returns = []
    entry_features = []
    done = False

    while not done:
        action = agent.select_action(obs, deterministic=True)
        shaped = env._shape_action(action)

        # Capture feature vector at every BUY_LIMIT
        if shaped == 1:
            entry_features.append({
                'obs':         obs.copy(),
                'price':       window[min(env._idx, len(window)-1)]['price'],
                'momentum_8h': float(obs[16]),
                'momentum_1d': float(obs[13]),
                'volatility':  float(obs[4]),
                'price_norm':  float(obs[0]),
            })

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        pv = info['portfolio_value']
        portfolio_values.append(pv)
        if len(portfolio_values) >= 2:
            prev = portfolio_values[-2]
            step_returns.append((pv - prev) / (prev + 1e-9))

    if env._ob.position:
        env._ob.realize_gain(window[env._idx - 1]['price'])

    trades = env._ob.trades
    pv_arr = np.array(portfolio_values)
    ret_arr = np.array(step_returns) if step_returns else np.array([0.0])

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))

    return {
        'start_idx':      start_idx,
        'start_price':    window[MIN_WINDOW]['price'],
        'end_price':      window[-1]['price'],
        'market_return':  (window[-1]['price'] - window[MIN_WINDOW]['price']) / (window[MIN_WINDOW]['price'] + 1e-9) * 100,
        'total_return':   (pv_arr[-1] - env.initial_cash) / env.initial_cash * 100,
        'sortino':        compute_sortino(ret_arr),
        'max_drawdown':   compute_max_drawdown(pv_arr) * 100,
        'total_trades':   len(trades),
        'win_rate':       len(wins) / len(trades) * 100 if trades else 0.0,
        'profit_factor':  gp / gl if gl > 0 else float('inf'),
        'avg_hold':       np.mean([t['bars_held'] for t in trades]) if trades else 0.0,
        'entry_features': entry_features,
        'trades':         trades,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db',         default='/root/ticks.db')
    p.add_argument('--symbol',     default='BTCUSDT')
    p.add_argument('--checkpoint', default='checkpoints/btcusdt/nova_brain_best.pt')
    p.add_argument('--runs',       type=int, default=10)
    p.add_argument('--window',     type=int, default=10000,
                   help='Ticks per evaluation window (~7 days at 1-min)')
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    ticks = load_ticks_from_db(args.db, symbol=args.symbol)
    test_ticks = ticks[int(len(ticks) * 0.8):]
    print(f"\n[Consistency] {args.symbol} | {len(test_ticks):,} test ticks")
    print(f"[Consistency] {args.runs} runs × {args.window:,} tick windows")

    agent = SAC()
    agent.load(args.checkpoint)
    print(f"[Consistency] Checkpoint: step {agent.steps:,}  best_sortino {agent.best_sortino:.4f}\n")

    max_start = len(test_ticks) - args.window - 1
    starts = sorted(rng.integers(MIN_WINDOW, max_start, size=args.runs).tolist())

    results = []
    for i, start in enumerate(starts):
        r = run_window(agent, test_ticks, start, args.window, args.symbol)
        if r is None:
            continue
        results.append(r)
        trades_str = f"{r['total_trades']}T  {r['win_rate']:.0f}%WR" if r['total_trades'] > 0 else "0 trades"
        print(f"  Run {i+1:2d} | start={start:6,} | mkt={r['market_return']:+6.2f}% | "
              f"agent={r['total_return']:+6.2f}% | sortino={r['sortino']:+6.3f} | {trades_str}")

    if not results:
        print("No valid results.")
        return

    # Summary statistics
    sortinos     = [r['sortino'] for r in results]
    returns      = [r['total_return'] for r in results]
    trade_counts = [r['total_trades'] for r in results]
    win_rates    = [r['win_rate'] for r in results if r['total_trades'] > 0]
    drawdowns    = [r['max_drawdown'] for r in results]

    positive_runs = sum(1 for s in sortinos if s > 0)
    trading_runs  = sum(1 for t in trade_counts if t > 0)

    print(f"\n{'='*65}")
    print(f"  CONSISTENCY SUMMARY — {args.symbol}")
    print(f"{'='*65}")
    print(f"  Runs:              {len(results)}")
    print(f"  Positive Sortino:  {positive_runs}/{len(results)}  ({positive_runs/len(results)*100:.0f}%)")
    print(f"  Runs with trades:  {trading_runs}/{len(results)}")
    print(f"  Sortino:   mean={np.mean(sortinos):+.4f}  std={np.std(sortinos):.4f}  "
          f"min={np.min(sortinos):+.4f}  max={np.max(sortinos):+.4f}")
    print(f"  Return:    mean={np.mean(returns):+.4f}%  std={np.std(returns):.4f}%")
    print(f"  Drawdown:  mean={np.mean(drawdowns):.2f}%  worst={np.min(drawdowns):.2f}%")
    print(f"  Trades/run: mean={np.mean(trade_counts):.1f}  total={sum(trade_counts)}")
    if win_rates:
        print(f"  Win rate:  mean={np.mean(win_rates):.1f}%  (across runs with trades)")

    # Entry feature consistency — do entries share common features?
    all_entries = [e for r in results for e in r['entry_features']]
    if all_entries:
        print(f"\n  ── Entry Feature Consistency ({len(all_entries)} total entries) ──")
        m8h  = [e['momentum_8h'] for e in all_entries]
        m1d  = [e['momentum_1d'] for e in all_entries]
        vol  = [e['volatility']  for e in all_entries]
        pnrm = [e['price_norm']  for e in all_entries]
        print(f"  momentum_8h:  mean={np.mean(m8h):+.5f}  std={np.std(m8h):.5f}  "
              f"positive={sum(1 for v in m8h if v>0)}/{len(m8h)}")
        print(f"  momentum_1d:  mean={np.mean(m1d):+.5f}  std={np.std(m1d):.5f}  "
              f"positive={sum(1 for v in m1d if v>0)}/{len(m1d)}")
        print(f"  volatility:   mean={np.mean(vol):.5f}   std={np.std(vol):.5f}")
        print(f"  price_norm:   mean={np.mean(pnrm):.5f}  std={np.std(pnrm):.5f}")

        # Verdict
        mom_consistent = abs(np.mean(m8h)) > np.std(m8h) * 0.5
        print(f"\n  Verdict: momentum_8h signal {'CONSISTENT ✓' if mom_consistent else 'INCONSISTENT ✗'} across entries")
    else:
        print(f"\n  No entries captured across {len(results)} runs — agent chose cash in all windows.")
        print(f"  This itself is a signal: policy strongly prefers not trading.")

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
