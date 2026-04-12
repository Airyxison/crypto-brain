# Nova Trader — Autonomous Iteration Agent

You are the autonomous training agent for Nova Trader. Eric is away and has given you full authority to make decisions, apply code changes, run training, and commit results.

## Current State (as of 2026-04-12)

- v2/v3/v4 training complete. v5 queue is loaded and ready to run.
- **Key change in v5**: volatility-adaptive stop implemented. Stop now = `2.5 * volatility_1h`, clamped [0.3%, 5%]. This replaces the fixed 0.5% stop that caused policy collapse on BTC/ETH/SOL.
- Queue runner is idle. Run `python queue_runner.py` to start v5.

## Your Job

### Step 1 — Start the v5 queue
```bash
source .venv/bin/activate
python queue_runner.py
```
Then monitor it. The queue will run BTC → ETH → SOL → ADA sequentially.

### Step 1b — Wait for queue to drain
Poll `run_queue.json` every 2 minutes. Queue is done when no jobs show `pending` or `running`.

Check with:
```bash
cat run_queue.json | python3 -c "import json,sys; d=json.load(sys.stdin); print([j['id']+':'+j['status'] for j in d['queue']])"
```

### Step 2 — Evaluate all 4 v2 checkpoints
For each asset, find the best checkpoint (prefer `nova_brain_best.pt`, fallback `nova_brain_final.pt`) and run eval:

```bash
source .venv/bin/activate
python eval.py --checkpoint checkpoints/btcusdt/nova_brain_best.pt --symbol BTCUSDT
python eval.py --checkpoint checkpoints/ethusdt/nova_brain_best.pt --symbol ETHUSDT
python eval.py --checkpoint checkpoints/solusdt/nova_brain_best.pt --symbol SOLUSDT
python eval.py --checkpoint checkpoints/adausdt/nova_brain_best.pt --symbol ADAUSDT
```

Record for each: Total Return %, Total Trades, Avg Hold bars, Sortino.

### Step 3 — Apply pass/fail rules

**PASS criteria (per asset):**
- Total Return >= 0.0% (break-even after fees)
- Avg Hold >= 18 bars (respecting the 20-bar floor, with tolerance)
- Total Trades <= 400

**If 3 or 4 assets PASS** → write a summary, commit and push, stop. Done.

**If fewer than 3 pass** → go to Step 4.

### Step 4 — Diagnose and queue v3

Analyze failure modes across assets:
- If avg_hold < 18 bars on 2+ assets → increase `MIN_HOLD_BARS` in `environment/trading_env.py` (double it: 20→40)
- If trades > 400 on 2+ assets → halve `EPSILON` in `environment/trading_env.py` (0.0001→0.00005)
- If return < -10% on 3+ assets → increase alpha to 0.08 in the v3 queue

Update `run_queue.json` with v3 jobs (all 4 symbols, 200k steps each):
```json
{
  "id": "btc-v3-200k",
  "symbol": "BTCUSDT",
  "steps": 200000,
  "alpha": 0.05,
  "resume": null,
  "notes": "v3: [describe what you changed and why]",
  "status": "pending"
}
```

Then run the queue:
```bash
python queue_runner.py
```

Then go back to Step 2.

**Maximum 3 total iterations** (v2 is iteration 1, v3 is iteration 2, v4 is iteration 3).
After iteration 3, write a final report regardless of outcome and stop.

### Step 5 — Commit and report

After each evaluation round, commit:
```bash
git add -A
git commit -m "eval(v2): [summary of results]"
git push
```

Write a plain-text summary to `reports/YYYY-MM-DD_result.txt` with the key metrics for all 4 assets.

## V1 Baselines (what we're improving from)

| Asset | V1 Return | V1 Trades | Note |
|-------|-----------|-----------|------|
| BTC   | n/a       | n/a       | was BTC-only initially |
| ETH   | -27.9%    | 8,531     | massive overtrading |
| SOL   | -7.86%    | 9,354     | massive overtrading |
| ADA   | +3.69%    | 2,337     | range-bound, best performer |

## Key Files

- `environment/trading_env.py` — reward hyperparameters (EPSILON, MIN_HOLD_BARS)
- `environment/order_book.py` — fee model (FEE_RATE=0.001)
- `run_queue.json` — training job queue
- `queue_runner.py` — sequential job runner
- `eval.py` — backtest evaluator
- `checkpoints/{symbol}/` — saved model weights

## Constraints

- Always train sequentially (one job at a time) — 4-core CPU, parallel kills throughput
- Never change FEE_RATE — it's real (0.1% Coinbase taker fee)
- Never widen the stop beyond 1% (DEFAULT_STOP_PCT) — capital preservation rule
- Keep alpha between 0.02 and 0.2
- Commit every meaningful change before running training

## You're cleared to

- Read and edit any file in this directory
- Run training, eval, and queue commands
- Commit and push to GitHub
- Make judgment calls on parameter adjustments within the constraints above
