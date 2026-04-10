# PRD-001 — Fix best_sortino Reset Bug in train.py

**Status:** Ready for implementation  
**Priority:** P0 — Blocks all future training runs  
**Author:** Nova  
**Date:** 2026-04-10

---

## Problem

Every time `train.py` is launched with `--resume`, the `best_sortino` tracker is initialized to `-inf`:

```python
best_sortino = -np.inf   # line 72 of train.py
```

This means the very first validation backtest — typically at step 10k — will always produce a Sortino higher than `-inf`, causing `nova_brain_best.pt` to be overwritten with whatever the current (often poor) policy produces. The known-good checkpoint is destroyed within the first checkpoint interval.

### Evidence from iter_log.md

- **Run 5:** best_sortino bug overwrote nova_brain_best.pt at step 10k with Sortino -0.3347 (starting from 1.3186). Had to restore from `nova_brain_step40000.pt` backup.
- **Run 6:** Same bug overwrote nova_brain_best.pt at step 5k with Sortino -0.9141 (starting from 1.7294). `nova_brain_iter4_100k_sortino1.73.pt` was the only reason the checkpoint survived.

The current best checkpoint (`nova_brain_iter4_100k_sortino1.73.pt`) only survived because it was manually backed up with a timestamped filename. This is a fragile workflow that will eventually result in permanent loss of a good policy.

---

## Goal

When `--resume` is provided, initialize `best_sortino` from the Sortino value stored in (or inferred from) the checkpoint, so that `nova_brain_best.pt` is only overwritten when the policy genuinely improves.

---

## Acceptance Criteria

1. `python train.py --resume checkpoints/nova_brain_iter4_100k_sortino1.73.pt --steps 50000` does not overwrite `nova_brain_best.pt` unless a checkpoint with Sortino > 1.7294 is produced.
2. Fresh runs (no `--resume`) still initialize `best_sortino = -inf` (unchanged behavior).
3. The Sortino seed value is logged at startup so it is visible in training output.
4. No changes to the checkpoint file format are required for existing checkpoints — graceful fallback if the key is missing.

---

## Implementation Plan

### Option A — Store Sortino in checkpoint (preferred)

1. In `agent/sac.py`, add `'best_sortino'` to the `save()` dict:
   ```python
   torch.save({
       'actor':        self.actor.state_dict(),
       'critic1':      self.critic1.state_dict(),
       'critic2':      self.critic2.state_dict(),
       'alpha_value':  self.alpha_value,
       'steps':        self.steps,
       'best_sortino': self.best_sortino,   # NEW
   }, path)
   ```
2. Add `self.best_sortino = -np.inf` to `SAC.__init__`.
3. In `SAC.load()`, restore `self.best_sortino = ck.get('best_sortino', -np.inf)`.
4. In `train.py`, replace the hardcoded `best_sortino = -np.inf` with:
   ```python
   best_sortino = agent.best_sortino
   print(f"[TRAIN] best_sortino initialized to {best_sortino:.4f}")
   ```
5. After updating `best_sortino` in the checkpoint loop, sync it back:
   ```python
   agent.best_sortino = best_sortino
   ```

### Option B — Parse Sortino from checkpoint filename (simpler, no schema change)

Parse the Sortino value from filename patterns like `nova_brain_iter4_100k_sortino1.73.pt` or `nova_brain_step40000.pt`. Regex: `sortino([0-9.]+)`.

Fallback to `-inf` if pattern not found.

**Tradeoff:** Option A is authoritative and survives renames. Option B requires no changes to the checkpoint format and works for existing files. Recommend Option A for new checkpoints, with Option B as a fallback for legacy files.

---

## Files to Change

| File | Change |
|------|--------|
| `train.py` | Initialize `best_sortino` from `agent.best_sortino` after load |
| `agent/sac.py` | Add `best_sortino` field; save/load it in checkpoint dict |

---

## Testing

1. Save a checkpoint with a known Sortino (e.g., the real iter4 file at 1.7294).
2. Run `python train.py --resume <that checkpoint> --steps 10000 --save-every 5000`.
3. Confirm `nova_brain_best.pt` is not replaced at step 5k (Sortino will likely be < 1.7294).
4. Manually inject a higher Sortino by mock-patching `run_backtest` to return 2.0; confirm `nova_brain_best.pt` is updated.

---

## Notes

- Do not start any new training run until this fix is in place. Every resume-based run without this fix risks destroying the production policy.
- After implementing, update the Sortino saved in `nova_brain_iter4_100k_sortino1.73.pt` by loading and re-saving it so it carries the `best_sortino` key going forward.
