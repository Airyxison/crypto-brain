"""
Historical Candle Backfill
--------------------------
Fetches BTC-USD 1-minute candles from the Coinbase Exchange public REST API
and writes them into ticks.db so the training pipeline has rich history.

One candle → one tick (close price, candle volume).
The feature engineer windows are tick-based, so this is semantically equivalent
to 1-minute resolution data — labels like "1m momentum" become "60-candle momentum."

Usage:
  python backfill.py --days 90                              # BTC, last 90 days
  python backfill.py --days 365                             # BTC, full year
  python backfill.py --product ETH-USD --symbol ETHUSDT --days 365
  python backfill.py --product SOL-USD --symbol SOLUSDT --days 365
  python backfill.py --days 30 --dry-run                    # preview without writing

Coinbase public API: no auth required, 10 req/s limit (we use ~3 req/s).
"""

import sqlite3
import time
import argparse
from datetime import datetime, timezone, timedelta

import requests


COINBASE_API  = "https://api.exchange.coinbase.com"
GRANULARITY   = 60    # 1-minute candles
MAX_PER_REQ   = 300   # Coinbase returns max 300 candles per call
SLEEP_BETWEEN = 0.35  # seconds between requests (~3/s, well under 10/s limit)


def fetch_candles(start: datetime, end: datetime, product_id: str) -> list:
    """
    Returns list of [timestamp, low, high, open, close, volume] sorted oldest-first.
    Coinbase returns newest-first so we reverse.
    """
    url    = f"{COINBASE_API}/products/{product_id}/candles"
    params = {
        'granularity': GRANULARITY,
        'start':       start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end':         end.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    candles = resp.json()
    return list(reversed(candles))  # oldest first


def candles_to_ticks(candles: list, symbol: str) -> list[dict]:
    ticks = []
    for c in candles:
        ts, low, high, open_, close, volume = c
        ticks.append({
            'symbol':     symbol,
            'price':      float(close),
            'quantity':   float(volume),
            'trade_time': int(ts) * 1000,   # ms, matching Rust engine format
        })
    return ticks


def write_ticks(conn: sqlite3.Connection, ticks: list[dict]) -> int:
    cur = conn.cursor()
    cur.executemany(
        """INSERT OR IGNORE INTO ticks (symbol, price, quantity, trade_time)
           VALUES (:symbol, :price, :quantity, :trade_time)""",
        ticks,
    )
    conn.commit()
    return cur.rowcount  # rows actually inserted (skips duplicates)


def get_existing_range(conn: sqlite3.Connection, symbol: str) -> tuple[int | None, int | None]:
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(trade_time), MAX(trade_time) FROM ticks WHERE symbol = ?",
        (symbol,),
    )
    row = cur.fetchone()
    return row[0], row[1]  # (min_ms, max_ms) or (None, None)


def main():
    p = argparse.ArgumentParser(description='Backfill ticks.db with Coinbase historical candles')
    p.add_argument('--db',       default='../crypto-engine/ticks.db',
                   help='Path to SQLite ticks database')
    p.add_argument('--days',     type=int, default=90,
                   help='Days of history to fetch (default: 90)')
    p.add_argument('--product',  default='BTC-USD',
                   help='Coinbase product ID to fetch (default: BTC-USD, e.g. ETH-USD, SOL-USD)')
    p.add_argument('--symbol',   default=None,
                   help='Symbol to write into DB — defaults to product with hyphen removed (e.g. BTC-USD → BTCUSD)')
    p.add_argument('--dry-run',  action='store_true',
                   help='Fetch but do not write to DB')
    args = p.parse_args()

    # Derive DB symbol from product ID if not explicitly provided (BTC-USD → BTCUSD)
    if args.symbol is None:
        args.symbol = args.product.replace('-', '')

    conn = sqlite3.connect(args.db)

    # Ensure table exists with a unique constraint so INSERT OR IGNORE deduplicates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     TEXT    NOT NULL,
            price      REAL    NOT NULL,
            quantity   REAL    NOT NULL,
            trade_time INTEGER NOT NULL,
            UNIQUE(symbol, trade_time)
        )
    """)
    conn.commit()

    # Show what's already in the DB
    existing_min, existing_max = get_existing_range(conn, args.symbol)
    if existing_min:
        dt_min = datetime.fromtimestamp(existing_min / 1000, tz=timezone.utc)
        dt_max = datetime.fromtimestamp(existing_max / 1000, tz=timezone.utc)
        print(f"[BACKFILL] Existing data: {dt_min.strftime('%Y-%m-%d %H:%M')} → {dt_max.strftime('%Y-%m-%d %H:%M')} UTC")
    else:
        print("[BACKFILL] No existing data found for this symbol.")

    end   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    chunk_size = timedelta(seconds=GRANULARITY * MAX_PER_REQ)  # 300 minutes
    total_fetched  = 0
    total_written  = 0
    total_skipped  = 0
    current        = start
    chunk_num      = 0
    total_chunks   = int((end - start) / chunk_size) + 1

    print(f"[BACKFILL] Fetching {args.days} days of {args.product} 1m candles → symbol='{args.symbol}'")
    print(f"[BACKFILL] Range: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} UTC")
    print(f"[BACKFILL] {total_chunks} API requests needed (~{total_chunks * SLEEP_BETWEEN:.0f}s)")
    if args.dry_run:
        print("[BACKFILL] DRY RUN — no writes to DB")
    print()

    while current < end:
        chunk_end = min(current + chunk_size, end)
        chunk_num += 1

        try:
            candles = fetch_candles(current, chunk_end, args.product)
            ticks   = candles_to_ticks(candles, args.symbol)
            total_fetched += len(ticks)

            if ticks and not args.dry_run:
                written = write_ticks(conn, ticks)
                skipped = len(ticks) - written
                total_written += written
                total_skipped += skipped
            else:
                written = len(ticks)
                skipped = 0

            price_lo = min(t['price'] for t in ticks) if ticks else 0
            price_hi = max(t['price'] for t in ticks) if ticks else 0
            print(
                f"  [{chunk_num:>4}/{total_chunks}] {current.strftime('%Y-%m-%d %H:%M')} "
                f"→ {len(ticks):3d} candles  "
                f"price ${price_lo:,.0f}–${price_hi:,.0f}  "
                f"{'(skipped ' + str(skipped) + ' dups)' if skipped else ''}"
            )

        except requests.exceptions.HTTPError as e:
            print(f"  [{chunk_num:>4}/{total_chunks}] HTTP {e.response.status_code} — skipping chunk, retrying next")
            time.sleep(2.0)
        except Exception as e:
            print(f"  [{chunk_num:>4}/{total_chunks}] ERROR: {e} — skipping chunk")

        current = chunk_end
        time.sleep(SLEEP_BETWEEN)

    conn.close()

    print()
    print(f"[BACKFILL] Done.")
    print(f"  Fetched:  {total_fetched:,} candles")
    if not args.dry_run:
        print(f"  Written:  {total_written:,} new ticks")
        print(f"  Skipped:  {total_skipped:,} duplicates")
        print(f"  DB path:  {args.db}")
    print()
    print(f"  Next: python -u train.py --symbol {args.symbol} --steps 200000")


if __name__ == '__main__':
    main()
