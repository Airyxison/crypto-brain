"""
Eval Script — Nova Brain
------------------------
Run a backtest on any saved checkpoint without touching the training loop.

Usage:
  python eval.py --checkpoint checkpoints/nova_brain_best.pt
  python eval.py --checkpoint checkpoints/nova_brain_final.pt --days 30
  python eval.py --checkpoint checkpoints/nova_brain_best.pt --plot
"""

import argparse
import numpy as np
from pathlib import Path

from backtest.runner import load_ticks_from_db, run_backtest
from agent.sac import SAC


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a Nova Brain checkpoint')
    p.add_argument('--checkpoint', required=True,          help='Path to .pt checkpoint')
    p.add_argument('--db',         default='../crypto-engine/ticks.db', help='SQLite tick DB')
    p.add_argument('--symbol',     default='BTCUSDT',      help='Asset symbol')
    p.add_argument('--days',       type=int, default=None, help='Limit to last N days of data')
    p.add_argument('--split',      type=float, default=0.2,help='Fraction of data to use as eval set (from end)')
    p.add_argument('--plot',       action='store_true',    help='Plot portfolio curve (requires matplotlib)')
    return p.parse_args()


def main():
    args = parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        print(f"[EVAL] Checkpoint not found: {ck_path}")
        return

    print(f"[EVAL] Loading checkpoint: {ck_path}")
    agent = SAC()
    agent.load(str(ck_path))

    print(f"[EVAL] Loading ticks from {args.db}...")
    ticks = load_ticks_from_db(args.db, symbol=args.symbol)
    print(f"[EVAL] {len(ticks):,} ticks loaded")

    if len(ticks) < 500:
        print("[EVAL] Need at least 500 ticks.")
        return

    # Optionally restrict to last N days
    if args.days:
        cutoff_ms = ticks[-1]['trade_time'] - args.days * 24 * 3600 * 1000
        ticks = [t for t in ticks if t['trade_time'] >= cutoff_ms]
        print(f"[EVAL] Filtered to last {args.days} days: {len(ticks):,} ticks")

    # Use the tail fraction as the eval set (matches train/test split logic)
    split = int(len(ticks) * (1.0 - args.split))
    eval_ticks = ticks[split:]
    print(f"[EVAL] Eval set: {len(eval_ticks):,} ticks ({args.split*100:.0f}% of data)\n")

    results = run_backtest(agent, eval_ticks, verbose=True)

    # Trade-level breakdown
    trades = results.get('trades', [])
    if trades:
        wins  = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        print(f"  Trade breakdown:")
        print(f"    Winners: {len(wins)}  avg P&L: {np.mean([t['pnl_pct']*100 for t in wins]):.2f}%" if wins else "    Winners: 0")
        print(f"    Losers:  {len(losses)}  avg P&L: {np.mean([t['pnl_pct']*100 for t in losses]):.2f}%" if losses else "    Losers: 0")
        reasons = {}
        for t in trades:
            reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
        print(f"    Exit reasons: {reasons}")
        print()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            curve = np.array(results['portfolio_curve'])
            plt.figure(figsize=(12, 4))
            plt.plot(curve, linewidth=1)
            plt.axhline(y=10_000, color='gray', linestyle='--', alpha=0.5)
            plt.title(f"Portfolio Curve — {ck_path.name}")
            plt.xlabel("Steps")
            plt.ylabel("Portfolio Value ($)")
            plt.tight_layout()
            out = ck_path.with_suffix('.png')
            plt.savefig(out, dpi=150)
            print(f"[EVAL] Plot saved → {out}")
        except ImportError:
            print("[EVAL] matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
