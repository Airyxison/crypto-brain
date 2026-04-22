"""
Regime Distribution Analysis
-----------------------------
For each symbol, walks the full training split chronologically and classifies
every bar into a regime using the same logic as trading_env.py.

Reports:
  - Regime distribution (% BULL / BEAR / RANGE / HIGH_VOL)
  - Avg momentum_8h magnitude per regime (how strong the signal is)
  - Avg price_move per regime (how actionable the opp_cost is)
  - Momentum_8h autocorrelation (how persistent/reliable the signal is)
  - Recommended EPSILON scaling per symbol (relative to BTC baseline)
"""

import sqlite3
import argparse
import numpy as np
from collections import deque

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
REGIME_WINDOW = 480       # 8h in 1-min bars
VOL_WINDOW    = 60        # 1h
TRAIN_RATIO   = 0.8

# Regime thresholds (match trading_env.py)
BULL_MOMENTUM_THRESH = 0.0
BEAR_MOMENTUM_THRESH = 0.0


def classify_regime(momentum_8h: float, vol_ratio: float) -> str:
    if vol_ratio > 2.0:
        return "HIGH_VOL"
    if momentum_8h > BULL_MOMENTUM_THRESH:
        return "BULL"
    if momentum_8h < BEAR_MOMENTUM_THRESH:
        return "BEAR"
    return "RANGE"


def analyze_symbol(db_path: str, symbol: str) -> dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT price, quantity FROM ticks WHERE symbol=? ORDER BY trade_time ASC",
        (symbol,)
    ).fetchall()
    conn.close()

    n_train = int(len(rows) * TRAIN_RATIO)
    train = rows[:n_train]
    print(f"\n{symbol}: {len(train):,} training bars")

    prices  = deque(maxlen=REGIME_WINDOW + 1)
    vols    = deque(maxlen=VOL_WINDOW)
    returns = []

    regime_counts   = {"BULL": 0, "BEAR": 0, "RANGE": 0, "HIGH_VOL": 0}
    momentum_by_reg = {"BULL": [], "BEAR": [], "RANGE": [], "HIGH_VOL": []}
    move_by_reg     = {"BULL": [], "BEAR": [], "RANGE": [], "HIGH_VOL": []}
    momentum_series = []

    for i, (price, quantity) in enumerate(train):
        prices.append(price)
        vols.append(price)

        if i < 2:
            continue

        prev_price = train[i-1][0]
        price_move = (price - prev_price) / (prev_price + 1e-9)
        returns.append(price_move)

        if len(prices) < REGIME_WINDOW:
            continue

        momentum_8h = (prices[-1] - prices[-REGIME_WINDOW]) / (prices[-REGIME_WINDOW] + 1e-9)

        # vol ratio: current 1h std vs rolling mean
        if len(returns) >= VOL_WINDOW * 2:
            recent_vol  = float(np.std(returns[-VOL_WINDOW:]))
            baseline_vol = float(np.std(returns[-VOL_WINDOW*2:-VOL_WINDOW]))
            vol_ratio = recent_vol / (baseline_vol + 1e-9)
        else:
            vol_ratio = 1.0

        regime = classify_regime(momentum_8h, vol_ratio)
        regime_counts[regime] += 1
        momentum_by_reg[regime].append(abs(momentum_8h))
        move_by_reg[regime].append(abs(price_move))
        momentum_series.append(momentum_8h)

    total = sum(regime_counts.values())

    # Autocorrelation of momentum_8h at lag 1 (persistence)
    if len(momentum_series) > 1:
        m = np.array(momentum_series)
        autocorr = float(np.corrcoef(m[:-1], m[1:])[0, 1])
    else:
        autocorr = 0.0

    # Typical move size in BULL/BEAR (what EPSILON is amplifying)
    bull_move = float(np.mean(move_by_reg["BULL"])) if move_by_reg["BULL"] else 0.0
    bear_move = float(np.mean(move_by_reg["BEAR"])) if move_by_reg["BEAR"] else 0.0
    actionable_move = (bull_move + bear_move) / 2.0

    return {
        "symbol":         symbol,
        "total_bars":     total,
        "regime_pct":     {k: 100 * v / total for k, v in regime_counts.items()},
        "avg_mom_mag":    {k: float(np.mean(v)) if v else 0.0 for k, v in momentum_by_reg.items()},
        "avg_move":       {k: float(np.mean(v)) if v else 0.0 for k, v in move_by_reg.items()},
        "momentum_autocorr": autocorr,
        "actionable_move":   actionable_move,
    }


def print_report(results: list):
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION ANALYSIS — v12.5 Training Data")
    print("="*70)

    btc = next(r for r in results if r["symbol"] == "BTCUSDT")
    btc_move = btc["actionable_move"]

    for r in results:
        sym   = r["symbol"]
        pct   = r["regime_pct"]
        mom   = r["avg_mom_mag"]
        move  = r["avg_move"]
        acorr = r["momentum_autocorr"]
        eps_scale = r["actionable_move"] / (btc_move + 1e-9)

        print(f"\n{'─'*50}")
        print(f"  {sym}  ({r['total_bars']:,} bars)")
        print(f"{'─'*50}")
        print(f"  Regime distribution:")
        print(f"    BULL     {pct['BULL']:6.1f}%   avg mom={mom['BULL']:.4f}  avg move={move['BULL']:.5f}")
        print(f"    BEAR     {pct['BEAR']:6.1f}%   avg mom={mom['BEAR']:.4f}  avg move={move['BEAR']:.5f}")
        print(f"    RANGE    {pct['RANGE']:6.1f}%   avg mom={mom['RANGE']:.4f}  avg move={move['RANGE']:.5f}")
        print(f"    HIGH_VOL {pct['HIGH_VOL']:6.1f}%")
        print(f"  Momentum_8h autocorr (persistence): {acorr:.4f}")
        print(f"  Avg actionable move (BULL+BEAR):    {r['actionable_move']:.5f}")
        print(f"  Recommended EPSILON scale vs BTC:   {eps_scale:.3f}x")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        trending = r["regime_pct"]["BULL"] + r["regime_pct"]["BEAR"]
        eps_scale = r["actionable_move"] / (btc_move + 1e-9)
        print(f"  {r['symbol']:10s}  trending={trending:5.1f}%  "
              f"autocorr={r['momentum_autocorr']:.3f}  eps_scale={eps_scale:.3f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/root/ticks.db")
    parser.add_argument("--symbol", default=None, help="Single symbol or all if omitted")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    results = [analyze_symbol(args.db, sym) for sym in symbols]
    print_report(results)
