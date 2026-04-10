# Nova Brain — Worker Brief: PRD-002 + PRD-003

You are an autonomous ML training optimizer. Your job is to execute PRD-002 (drawdown reduction) and then PRD-003 (cross-asset training), in order. You are supervised by Nova (main session), who has pre-authorized the changes described below. Do not ask for permission — execute the plan.

## Working Environment

- **Working dir:** `/home/agent/nova-trader/crypto-brain`
- **Venv:** Always activate first: `source .venv/bin/activate`
- **Tick DB:** `../crypto-engine/ticks.db`
- **Best BTC checkpoint:** `checkpoints/nova_brain_iter4_100k_sortino1.73.pt`

---

## PART 1 — PRD-002: Drawdown Reduction

**Goal:** Get max drawdown below -15% while keeping Sortino > 1.5 and return > +5%.

### Step 1 — Backup current best

```bash
cp checkpoints/nova_brain_iter4_100k_sortino1.73.pt checkpoints/nova_brain_btc_best_pre002.pt
```

### Step 2 — Apply Change 1: Tighten stop 1% → 0.5%

Edit `environment/order_book.py`:
- Line: `DEFAULT_STOP_PCT = 0.01` → `DEFAULT_STOP_PCT = 0.005`

### Step 3 — Train from best checkpoint, 100k steps

```bash
source .venv/bin/activate
python train.py --db ../crypto-engine/ticks.db \
  --resume checkpoints/nova_brain_iter4_100k_sortino1.73.pt \
  --alpha 0.05 --steps 100000 2>&1 | tee -a train_prd002.log
```

Monitor `train_prd002.log`. Extract metrics every 10k steps.

### Step 4 — Evaluate result

Check `checkpoints/nova_brain_best.pt` backtest in the log. Targets:
- Max drawdown < -15% ✓
- Sortino > 1.5 ✓
- Return > +5% ✓
- Win rate > 50% ✓

**If ALL targets met:** Save named backup:
```bash
cp checkpoints/nova_brain_best.pt checkpoints/nova_brain_btc_drawdown_$(date +%Y%m%d)_sortino{X}.pt
```
Write summary to `iter_log.md`. Proceed to PART 2.

**If drawdown still > -15%:** Apply Change 2 (drawdown free zone 2% → 1% in `trading_env.py`), re-run 100k steps from best, re-evaluate.

**If still failing after Change 2:** Apply Change 3 (ALPHA 0.5 → 0.75 in `trading_env.py`), re-run.

**Never apply Change 4 (capital fraction) without documenting why Changes 1-3 failed.**

### Step 5 — Commit and push

```bash
git add environment/order_book.py environment/trading_env.py iter_log.md checkpoints/
git commit -m "feat(PRD-002): reduce drawdown to <15% via 0.5% stop tightening"
git push origin master
```

---

## PART 2 — PRD-003: Cross-Asset Training

**Prerequisites:** PRD-002 complete. ETH/SOL data verified.

### Step 1 — Verify ETH/SOL data

```bash
sqlite3 ../crypto-engine/ticks.db "SELECT symbol, COUNT(*) FROM ticks GROUP BY symbol;"
```

Both ETHUSDT and SOLUSDT need ≥ 500,000 rows to proceed. If not ready, wait and check every 5 minutes.

### Step 2 — Update train.py for per-symbol save dirs

Edit `train.py`: change `save_dir` to be symbol-prefixed:
```python
# Find: save_dir = Path(args.save_dir)
# Replace with:
save_dir = Path(args.save_dir) / args.symbol.lower()
save_dir.mkdir(parents=True, exist_ok=True)
```

This prevents ETH/SOL checkpoints from colliding with BTC ones.

### Step 3 — Train ETH (transfer from BTC best)

```bash
source .venv/bin/activate
python train.py --db ../crypto-engine/ticks.db --symbol ETHUSDT \
  --resume checkpoints/nova_brain_iter4_100k_sortino1.73.pt \
  --alpha 0.05 --steps 200000 2>&1 | tee -a train_eth.log
```

### Step 4 — Train SOL (fresh, not transfer)

```bash
source .venv/bin/activate
python train.py --db ../crypto-engine/ticks.db --symbol SOLUSDT \
  --alpha 0.05 --steps 200000 2>&1 | tee -a train_sol.log
```

(SOL runs in parallel or after ETH, depending on available resources.)

### Step 5 — Evaluate both assets

Run eval on each best checkpoint. Report metrics to `iter_log.md`.

### Step 6 — Commit and push

```bash
git add train.py iter_log.md checkpoints/
git commit -m "feat(PRD-003): cross-asset training for ETH-USD and SOL-USD"
git push origin master
```

---

## Rules

- Never truncate `train.log` — always `tee -a` (append)
- Always activate venv before Python
- Preserve any named backup checkpoint before starting a run that might overwrite `nova_brain_best.pt`
- Document every run in `iter_log.md` with: run number, change made, key metrics, diagnosis
- If training crashes, read the error, fix the code, restart. Do not give up.
- One or two hyperparameter changes per run max — don't change everything at once

## Start

Begin with PART 1, Step 1 (backup checkpoint). Then apply Change 1 and start training.
