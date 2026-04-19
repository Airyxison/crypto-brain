"""
Backtest Runner
---------------
Runs the trained agent against a held-out slice of historical tick data
and computes performance metrics.

Metrics:
  - Total return %
  - Sortino ratio (annualized, downside-only risk)
  - Max drawdown %
  - Win rate (% of closed trades that were profitable)
  - Avg hold time (bars)
  - Total trades
  - P&L curve (portfolio value over time)
"""

import sqlite3
import numpy as np
from pathlib import Path

from environment.trading_env import TradingEnv
from agent.sac import SAC


def load_ticks_from_db(db_path: str, symbol: str = 'BTCUSDT', limit: int = 2_000_000) -> list[dict]:
    """Load historical ticks from the Rust engine's SQLite database."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(
        "SELECT price, quantity, trade_time FROM ticks WHERE symbol = ? ORDER BY trade_time ASC LIMIT ?",
        (symbol, limit)
    )
    rows = cur.fetchall()
    conn.close()

    return [{'price': r[0], 'volume': r[1], 'trade_time': r[2]} for r in rows]


def compute_sortino(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 525_600) -> float:
    """Annualized Sortino ratio. periods_per_year = minutes in a year for 1-min data."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    mean_return = np.mean(excess)
    return float((mean_return / downside_std) * np.sqrt(periods_per_year))


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak) / (peak + 1e-9)
    return float(drawdowns.min())


DEPLOYMENT_BAR = {
    'sortino_ratio':    0.05,   # must be >= this
    'max_drawdown_pct': -15.0,  # must be >= this (less negative)
    'win_rate_pct':     50.0,   # must be >= this
    'profit_factor':    1.2,    # must be >= this
}


def check_deployment_gate(results: dict, bar: dict | None = None) -> dict:
    """Return per-metric pass/fail and overall verdict against deployment thresholds."""
    thresholds = bar or DEPLOYMENT_BAR
    checks = {
        'sortino_ratio':    results['sortino_ratio']    >= thresholds['sortino_ratio'],
        'max_drawdown_pct': results['max_drawdown_pct'] >= thresholds['max_drawdown_pct'],
        'win_rate_pct':     results['win_rate_pct']     >= thresholds['win_rate_pct'],
        'profit_factor':    results.get('profit_factor', 0) >= thresholds['profit_factor'],
    }
    return {'checks': checks, 'passed': all(checks.values()), 'thresholds': thresholds}


def run_backtest(
    agent:    SAC,
    ticks:    list[dict],
    config:   dict | None = None,
    verbose:  bool = True,
    gate:     bool = False,
) -> dict:
    # Override episode truncation for evaluation — run the full tick slice,
    # not the short 200-step windows used during training.
    eval_config = dict(config or {})
    eval_config['max_episode_steps'] = len(ticks)

    from features.engineer import FeatureEngineer, MIN_WINDOW
    from environment.order_book import OrderBookSimulator

    env = TradingEnv(ticks, eval_config)

    # Deterministic reset: always start at MIN_WINDOW with a cleanly warmed
    # FeatureEngineer — avoids the random-start / feature-state mismatch.
    env._idx        = MIN_WINDOW
    env._step_count = 0
    env._ob         = OrderBookSimulator(env.initial_cash)
    env._features   = FeatureEngineer()
    for i in range(MIN_WINDOW):
        t = ticks[i]
        env._features.update(t['price'], t['volume'], t['trade_time'])
    obs = env._get_obs()

    portfolio_values = [env._ob.portfolio_value]
    step_returns     = []
    action_counts    = [0] * 5
    done = False

    while not done:
        action = agent.select_action(obs, deterministic=True)
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        pv = info['portfolio_value']
        portfolio_values.append(pv)

        if len(portfolio_values) >= 2:
            prev = portfolio_values[-2]
            step_returns.append((pv - prev) / (prev + 1e-9))

    # Force-close any open position at last price so it registers as a trade
    if env._ob.position:
        last_price = ticks[env._idx - 1]['price']
        env._ob.realize_gain(last_price)

    pv_arr     = np.array(portfolio_values)
    ret_arr    = np.array(step_returns) if step_returns else np.array([0.0])
    trades     = env._ob.trades
    initial_pv = env.initial_cash

    total_return  = (pv_arr[-1] - initial_pv) / initial_pv * 100
    max_drawdown  = compute_max_drawdown(pv_arr) * 100
    sortino       = compute_sortino(ret_arr)

    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades  = [t for t in trades if t['pnl'] <= 0]
    win_rate       = len(winning_trades) / len(trades) * 100 if trades else 0.0
    avg_hold       = np.mean([t['bars_held'] for t in trades]) if trades else 0.0
    gross_profit   = sum(t['pnl'] for t in winning_trades)
    gross_loss     = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor  = round(gross_profit / gross_loss, 4) if gross_loss > 0 else float('inf')

    results = {
        'total_return_pct':  round(total_return, 2),
        'sortino_ratio':     round(sortino, 4),
        'max_drawdown_pct':  round(max_drawdown, 2),
        'win_rate_pct':      round(win_rate, 2),
        'profit_factor':     profit_factor,
        'total_trades':      len(trades),
        'avg_hold_bars':     round(avg_hold, 1),
        'final_value':       round(pv_arr[-1], 2),
        'portfolio_curve':   pv_arr.tolist(),
        'trades':            trades,
        'action_counts':     action_counts,
    }

    ACTION_NAMES = ['HOLD', 'BUY', 'ADJ_STOP', 'REALIZE', 'CANCEL']
    total_a = max(sum(action_counts), 1)
    action_dist = '  '.join(f"{ACTION_NAMES[i]}:{action_counts[i]/total_a*100:.0f}%" for i in range(5))

    if gate:
        results['deployment_gate'] = check_deployment_gate(results)

    if verbose:
        print("\n" + "="*50)
        print("  BACKTEST RESULTS")
        print("="*50)
        print(f"  Total Return:   {results['total_return_pct']:+.2f}%")
        print(f"  Sortino Ratio:  {results['sortino_ratio']:.4f}")
        print(f"  Max Drawdown:   {results['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:       {results['win_rate_pct']:.1f}%")
        print(f"  Profit Factor:  {results['profit_factor']:.4f}")
        print(f"  Total Trades:   {results['total_trades']}")
        print(f"  Avg Hold:       {results['avg_hold_bars']:.0f} bars")
        print(f"  Final Value:    ${results['final_value']:,.2f}")
        print(f"  Actions:        {action_dist}")
        if 'deployment_gate' in results:
            dg = results['deployment_gate']
            verdict = "PASS ✓" if dg['passed'] else "FAIL ✗"
            print(f"\n  Deployment Gate: {verdict}")
            for metric, passed in dg['checks'].items():
                mark = "✓" if passed else "✗"
                print(f"    {mark} {metric}: {results.get(metric, '?')} (threshold: {dg['thresholds'][metric]})")
        print("="*50 + "\n")

    return results
