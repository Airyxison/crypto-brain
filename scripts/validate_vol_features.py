"""
validate_vol_features.py — Barney v1 Tier 1 Validation
-------------------------------------------------------
Run BEFORE training with the 19-dim feature set. Validates the two new
volume flow features meet Barney's quality thresholds on real tick data.

Three checks:
  1. Distribution sanity — both features should be roughly centered near 0,
     not stuck at ±1 (which signals thin-tape floor dominating).
  2. Threshold precision — when vol_flow_30 < -0.3 AND price falls next bar,
     what fraction is true signal vs noise?  Target: >50% precision.
  3. Decorrelation — corr(vol_flow_240, momentum_8h) should be < 0.7.
     If it's ≥0.7 the 240-bar lag isn't adding independent information.

Usage:
  python scripts/validate_vol_features.py --db /root/ticks.db
  python scripts/validate_vol_features.py --db /root/ticks.db --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT
  python scripts/validate_vol_features.py --db /root/ticks.db --symbol BTCUSDT --sample 50000
"""

import argparse
import sys
import numpy as np

from backtest.runner import load_ticks_from_db
from features.engineer import FeatureEngineer, MIN_WINDOW

# Barney's pass/fail thresholds
DIST_MEAN_MAX       = 0.15   # |mean(vol_flow)| must be below this — near-zero center
DIST_CLAMP_MAX      = 0.30   # fraction of values == ±1.0 — if >30%, thin-tape floor is wrong
PRECISION_THRESHOLD = 0.3    # flow_30 < -0.3 trigger threshold
PRECISION_MIN       = 0.50   # min fraction of bearish triggers where next bar falls
DECORR_MAX          = 0.70   # max |corr(vol_flow_240, momentum_8h)| allowed

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']


def compute_features(ticks: list[dict], sample: int | None = None) -> dict:
    """
    Run FeatureEngineer over a tick list and return arrays for each feature.
    If sample is set, uses a random contiguous window after warmup.
    """
    if sample and len(ticks) > sample + MIN_WINDOW:
        rng = np.random.default_rng(42)
        max_start = len(ticks) - sample - 1
        start = int(rng.integers(MIN_WINDOW, max_start))
        ticks = ticks[start - MIN_WINDOW: start + sample]

    fe = FeatureEngineer()
    position_state = {
        'in_position': False,
        'entry_price': None,
        'stop_price':  None,
        'bars_held':   0,
        'max_hold':    300,
    }

    vol_flow_30s   = []
    vol_flow_240s  = []
    momentum_8hs   = []
    prices         = []

    for t in ticks:
        fe.update(t['price'], t['volume'], t['trade_time'])
        if fe.ready:
            state = fe.extract(position_state)
            vol_flow_30s.append(state[17])
            vol_flow_240s.append(state[18])
            momentum_8hs.append(state[16])
            prices.append(t['price'])

    return {
        'vol_flow_30':   np.array(vol_flow_30s,  dtype=np.float64),
        'vol_flow_240':  np.array(vol_flow_240s, dtype=np.float64),
        'momentum_8h':   np.array(momentum_8hs,  dtype=np.float64),
        'prices':        np.array(prices,        dtype=np.float64),
        'n':             len(vol_flow_30s),
    }


def check_distribution(arr: np.ndarray, name: str) -> bool:
    """Check 1: mean near zero, not stuck at clamps."""
    mean   = float(arr.mean())
    std    = float(arr.std())
    p_pos1 = float((arr >= 0.999).mean())   # fraction at +1
    p_neg1 = float((arr <= -0.999).mean())  # fraction at -1
    clamp_frac = p_pos1 + p_neg1

    ok_mean  = abs(mean) < DIST_MEAN_MAX
    ok_clamp = clamp_frac < DIST_CLAMP_MAX

    status = 'PASS' if (ok_mean and ok_clamp) else 'FAIL'
    print(f"    {name}: mean={mean:+.4f}  std={std:.4f}  "
          f"at+1={p_pos1:.1%}  at-1={p_neg1:.1%}  clamped={clamp_frac:.1%}  [{status}]")
    return ok_mean and ok_clamp


def check_threshold_precision(vol_flow_30: np.ndarray, prices: np.ndarray) -> bool:
    """
    Check 2: when vol_flow_30 < -PRECISION_THRESHOLD, does price fall next bar?
    Also check the symmetric bull side.
    """
    n = min(len(vol_flow_30), len(prices)) - 1  # -1 for next-bar lookahead

    bear_triggers = np.where(vol_flow_30[:n] < -PRECISION_THRESHOLD)[0]
    bull_triggers = np.where(vol_flow_30[:n] > PRECISION_THRESHOLD)[0]

    if len(bear_triggers) == 0:
        print(f"    Threshold precision: NO BEAR triggers (flow_30 < {-PRECISION_THRESHOLD}) — check data")
        return False

    next_moves = prices[1:n+1] - prices[:n]

    bear_correct = (next_moves[bear_triggers] < 0).sum()
    bear_prec    = bear_correct / len(bear_triggers)

    bull_correct = (next_moves[bull_triggers] > 0).sum() if len(bull_triggers) > 0 else 0
    bull_prec    = bull_correct / len(bull_triggers) if len(bull_triggers) > 0 else 0.0

    bear_ok = bear_prec >= PRECISION_MIN
    bull_ok = bull_prec >= PRECISION_MIN

    print(f"    Bear precision (flow<{-PRECISION_THRESHOLD}): "
          f"{bear_correct}/{len(bear_triggers)} = {bear_prec:.1%}  "
          f"['PASS' if bear_ok else 'FAIL']  "
          f"coverage={len(bear_triggers)/n:.1%}")
    print(f"    Bull precision (flow>{PRECISION_THRESHOLD}): "
          f"{bull_correct}/{len(bull_triggers)} = {bull_prec:.1%}  "
          f"['PASS' if bull_ok else 'FAIL']  "
          f"coverage={len(bull_triggers)/n:.1%}")

    return bear_ok and bull_ok


def check_decorrelation(vol_flow_240: np.ndarray, momentum_8h: np.ndarray) -> bool:
    """Check 3: vol_flow_240 must not be a proxy for momentum_8h (corr < 0.7)."""
    corr = float(np.corrcoef(vol_flow_240, momentum_8h)[0, 1])
    ok   = abs(corr) < DECORR_MAX
    status = 'PASS' if ok else 'FAIL'
    print(f"    corr(vol_flow_240, momentum_8h) = {corr:+.4f}  [{status}]  "
          f"(|corr| < {DECORR_MAX} required)")
    return ok


def validate_symbol(symbol: str, db_path: str, sample: int | None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {symbol}")
    print(f"{'='*60}")

    ticks = load_ticks_from_db(db_path, symbol=symbol)
    if len(ticks) < MIN_WINDOW + 1000:
        print(f"  [SKIP] Only {len(ticks)} ticks — need >{MIN_WINDOW + 1000}")
        return True  # not a failure, just not enough data

    print(f"  Ticks loaded: {len(ticks):,}")
    data = compute_features(ticks, sample=sample)
    print(f"  Feature samples: {data['n']:,}")

    print("\n  Check 1 — Distribution sanity:")
    ok1a = check_distribution(data['vol_flow_30'],  'vol_flow_30 ')
    ok1b = check_distribution(data['vol_flow_240'], 'vol_flow_240')

    print("\n  Check 2 — Threshold precision (vol_flow_30):")
    ok2 = check_threshold_precision(data['vol_flow_30'], data['prices'])

    print("\n  Check 3 — Decorrelation (vol_flow_240 vs momentum_8h):")
    ok3 = check_decorrelation(data['vol_flow_240'], data['momentum_8h'])

    passed = ok1a and ok1b and ok2 and ok3
    verdict = 'ALL CHECKS PASSED ✓' if passed else 'CHECKS FAILED ✗ — investigate before training'
    print(f"\n  Verdict: {verdict}")
    return passed


def main():
    p = argparse.ArgumentParser(description='Barney v1 Tier 1 validation for vol_flow features')
    p.add_argument('--db',      default='/root/ticks.db',  help='Path to ticks SQLite DB')
    p.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to validate')
    p.add_argument('--sample',  type=int,  default=None,
                   help='Max ticks to sample per symbol (default: all). Use 50000 for speed.')
    args = p.parse_args()

    print(f"\n[VALIDATE] Barney v1 Tier 1 — vol_flow_30 + vol_flow_240")
    print(f"[VALIDATE] DB: {args.db}")
    print(f"[VALIDATE] Symbols: {args.symbols}")
    if args.sample:
        print(f"[VALIDATE] Sample: {args.sample:,} ticks per symbol")

    results = {}
    for sym in args.symbols:
        try:
            results[sym] = validate_symbol(sym, args.db, args.sample)
        except Exception as e:
            print(f"\n[ERROR] {sym}: {e}")
            results[sym] = False

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for sym, ok in results.items():
        status = 'PASS' if ok else 'FAIL'
        print(f"  {sym:<12} [{status}]")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n  All symbols passed Tier 1. Clear to train v12.9.")
    else:
        print(f"\n  One or more symbols FAILED. Do not launch training until resolved.")
        sys.exit(1)


if __name__ == '__main__':
    main()
