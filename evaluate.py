"""
Evaluation & Visualization
---------------------------
Load a trained checkpoint, run a backtest, and plot results.

Usage:
  python evaluate.py --checkpoint checkpoints/nova_brain_best.pt --db ../crypto-engine/ticks.db
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from agent.sac import SAC
from backtest.runner import load_ticks_from_db, run_backtest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--db',         default='../crypto-engine/ticks.db')
    p.add_argument('--symbol',     default='BTCUSDT')
    p.add_argument('--test-pct',   type=float, default=0.2, help='Fraction of data to use as test set')
    p.add_argument('--output',     default='backtest_report.png')
    return p.parse_args()


def plot_results(results: dict, ticks: list[dict], output_path: str):
    pv_curve = np.array(results['portfolio_curve'])
    trades   = results['trades']
    prices   = np.array([t['price'] for t in ticks[:len(pv_curve)]])

    fig = plt.figure(figsize=(14, 10), facecolor='#111827')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    title_color = '#E5E7EB'
    label_color = '#9CA3AF'
    grid_color  = '#1F2937'
    line_color  = '#06B6D4'   # cyan
    green       = '#10B981'
    red         = '#EF4444'

    def style_ax(ax, title):
        ax.set_facecolor('#1F2937')
        ax.tick_params(colors=label_color)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.title.set_color(title_color)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, color=grid_color, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

    # 1. Portfolio value curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pv_curve, color=line_color, linewidth=1.2, label='Portfolio Value')
    ax1.axhline(pv_curve[0], color=label_color, linestyle='--', linewidth=0.8, alpha=0.6, label='Starting Capital')
    ax1.fill_between(range(len(pv_curve)), pv_curve[0], pv_curve,
                     where=(pv_curve >= pv_curve[0]), alpha=0.15, color=green)
    ax1.fill_between(range(len(pv_curve)), pv_curve[0], pv_curve,
                     where=(pv_curve < pv_curve[0]),  alpha=0.15, color=red)
    ax1.legend(facecolor='#1F2937', labelcolor=title_color, fontsize=9)
    style_ax(ax1, 'Portfolio Value Over Time')

    # 2. Drawdown curve
    ax2 = fig.add_subplot(gs[1, 0])
    peak = np.maximum.accumulate(pv_curve)
    drawdown_pct = (pv_curve - peak) / (peak + 1e-9) * 100
    ax2.fill_between(range(len(drawdown_pct)), drawdown_pct, 0, color=red, alpha=0.4)
    ax2.plot(drawdown_pct, color=red, linewidth=0.8)
    style_ax(ax2, 'Drawdown %')

    # 3. Trade P&L distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if trades:
        pnls = [t['pnl_pct'] * 100 for t in trades]
        colors = [green if p > 0 else red for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
        ax3.axhline(0, color=label_color, linewidth=0.8)
    style_ax(ax3, 'Trade P&L % per Trade')

    # 4. Metrics summary
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    metrics = [
        ('Total Return',  f"{results['total_return_pct']:+.2f}%"),
        ('Sortino Ratio', f"{results['sortino_ratio']:.4f}"),
        ('Max Drawdown',  f"{results['max_drawdown_pct']:.2f}%"),
        ('Win Rate',      f"{results['win_rate_pct']:.1f}%"),
        ('Total Trades',  str(results['total_trades'])),
        ('Avg Hold',      f"{results['avg_hold_bars']:.0f} bars"),
        ('Final Value',   f"${results['final_value']:,.2f}"),
    ]
    for i, (label, val) in enumerate(metrics):
        color = green if (label == 'Total Return' and results['total_return_pct'] > 0) else title_color
        ax4.text(0.05, 0.9 - i * 0.13, label, transform=ax4.transAxes,
                 color=label_color, fontsize=10)
        ax4.text(0.55, 0.9 - i * 0.13, val, transform=ax4.transAxes,
                 color=color, fontsize=10, fontweight='bold')
    style_ax(ax4, 'Performance Metrics')

    # 5. Hold time distribution
    ax5 = fig.add_subplot(gs[2, 1])
    if trades:
        holds = [t['bars_held'] for t in trades]
        ax5.hist(holds, bins=20, color=line_color, alpha=0.7, edgecolor=grid_color)
    style_ax(ax5, 'Hold Duration Distribution (bars)')

    fig.suptitle('Nova Brain — Backtest Report', color=title_color, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[EVAL] Report saved → {output_path}")
    plt.close()


def main():
    args = parse_args()

    print(f"[EVAL] Loading ticks from {args.db}...")
    ticks = load_ticks_from_db(args.db)

    split = int(len(ticks) * (1.0 - args.test_pct))
    test_ticks = ticks[split:]
    print(f"[EVAL] Evaluating on {len(test_ticks)} test ticks")

    agent = SAC()
    agent.load(args.checkpoint)

    results = run_backtest(agent, test_ticks, verbose=True)

    # Save JSON report
    json_path = Path(args.output).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'portfolio_curve'}, f, indent=2)
    print(f"[EVAL] JSON report → {json_path}")

    plot_results(results, test_ticks, args.output)


if __name__ == '__main__':
    main()
