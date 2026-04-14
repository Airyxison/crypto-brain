"""
fetch_history.py
----------------
Fetches historical 1-minute klines from Binance and stores them
in a SQLite database compatible with the existing backtest runner.

Usage:
  python fetch_history.py --symbol SOLUSDT --years 3 --output /root/crypto-engine/sol_history.db

The output DB uses the same schema as ticks.db so it drops straight
into load_ticks_from_db() and run_backtest() with no changes.

Resumable: re-running skips candles already present (UNIQUE constraint).
"""

import argparse
import sqlite3
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm


BINANCE_KLINES = 'https://api.binance.com/api/v3/klines'
LIMIT          = 1000   # max candles per request (Binance cap)
INTERVAL       = '1m'


def create_db(db_path: str):
    conn = sqlite3.connect(db_path)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON ticks(symbol, trade_time)")
    conn.commit()
    return conn


def last_stored_time(conn, symbol: str) -> int | None:
    """Return the most recent trade_time stored for this symbol, or None."""
    cur = conn.execute(
        "SELECT MAX(trade_time) FROM ticks WHERE symbol = ?", (symbol,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch one batch of klines. Returns list of rows or raises on error."""
    resp = requests.get(BINANCE_KLINES, params={
        'symbol':    symbol,
        'interval':  INTERVAL,
        'startTime': start_ms,
        'endTime':   end_ms,
        'limit':     LIMIT,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def kline_to_row(symbol: str, k: list) -> tuple:
    """Convert Binance kline array to (symbol, price, quantity, trade_time).
    Uses close price and total volume for the 1-min candle."""
    open_time  = int(k[0])   # ms — start of candle, matches tick schema
    close_px   = float(k[4]) # close price
    volume     = float(k[5]) # base asset volume
    return (symbol, close_px, volume, open_time)


def main():
    p = argparse.ArgumentParser(description='Fetch Binance historical klines into SQLite')
    p.add_argument('--symbol',  default='SOLUSDT',   help='Trading pair e.g. SOLUSDT')
    p.add_argument('--years',   type=float, default=3.0, help='Years of history to fetch')
    p.add_argument('--output',  default='/root/crypto-engine/sol_history.db', help='Output SQLite path')
    p.add_argument('--end',     default=None, help='End date YYYY-MM-DD (default: now)')
    args = p.parse_args()

    # Time range
    now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms  = now_ms if not args.end else int(
        datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000
    )
    start_ms = end_ms - int(args.years * 365.25 * 24 * 60 * 60 * 1000)

    total_minutes = int((end_ms - start_ms) / 60_000)
    total_batches = (total_minutes + LIMIT - 1) // LIMIT

    print(f"[FETCH] {args.symbol} | {args.years}y | {total_minutes:,} candles | {total_batches:,} requests")
    print(f"[FETCH] {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}")
    print(f"[FETCH] Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(args.output)

    # Resume from last stored candle if any
    last = last_stored_time(conn, args.symbol)
    if last:
        resume_from = last + 60_000  # next minute after last stored
        skipped = int((resume_from - start_ms) / 60_000)
        print(f"[FETCH] Resuming from {datetime.fromtimestamp(last/1000, tz=timezone.utc)} ({skipped:,} candles already stored)")
        start_ms = resume_from

    cursor_ms = start_ms
    inserted  = 0
    errors    = 0

    with tqdm(total=total_batches, unit='batch', ncols=80) as pbar:
        while cursor_ms < end_ms:
            batch_end = min(cursor_ms + LIMIT * 60_000, end_ms)

            try:
                klines = fetch_klines(args.symbol, cursor_ms, batch_end)
            except requests.exceptions.RequestException as e:
                errors += 1
                print(f"\n[FETCH] Request error: {e} — retrying in 5s")
                time.sleep(5)
                continue

            if not klines:
                cursor_ms = batch_end + 60_000
                pbar.update(1)
                continue

            rows = [kline_to_row(args.symbol, k) for k in klines]
            try:
                conn.executemany(
                    "INSERT OR IGNORE INTO ticks (symbol, price, quantity, trade_time) VALUES (?,?,?,?)",
                    rows
                )
                conn.commit()
                inserted += len(rows)
            except sqlite3.Error as e:
                print(f"\n[FETCH] DB error: {e}")
                errors += 1

            cursor_ms = int(klines[-1][0]) + 60_000  # advance past last returned candle
            pbar.update(1)
            pbar.set_postfix(inserted=f'{inserted:,}', errors=errors)

            # Respect Binance rate limit — 1200 weight/min, klines = 2 weight each
            # 100ms sleep → ~600 req/min → well within limit
            time.sleep(0.1)

    conn.close()

    print(f"\n[FETCH] Done. {inserted:,} candles inserted, {errors} errors.")
    print(f"[FETCH] Verify: python3 -c \"import sqlite3; c=sqlite3.connect('{args.output}'); "
          f"print(c.execute('SELECT COUNT(*) FROM ticks').fetchone())\"")


if __name__ == '__main__':
    main()
