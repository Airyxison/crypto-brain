"""
migrate_checkpoint_17to19.py
-----------------------------
One-off migration script: extends a 17-dim SAC checkpoint to 19-dim.

Extends Actor and Critic first linear layers (net.0.weight) from [256,17]→[256,19].
Two new columns (vol_flow_30 [17] and vol_flow_240 [18]) initialized near-zero so
the existing policy is preserved at step 0. Gradients grow these columns gradually.

Usage:
    python scripts/migrate_checkpoint_17to19.py [--dry-run]
    python scripts/migrate_checkpoint_17to19.py --src checkpoints/btcusdt/nova_brain_best.pt \
                                                 --dst checkpoints/btcusdt/nova_brain_19dim.pt

Supports local paths (default) or S3 paths (s3://bucket/key).

Steps:
    1. Copy-tag source as <source>_pre_19dim_backup.pt  (protect baseline)
    2. Load source checkpoint
    3. Extend net.0.weight for Actor, Critic1, Critic2: [256,17] → [256,19]
    4. Assert shapes are correct
    5. Save migrated checkpoint to destination
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path

import torch


OLD_DIM = 17
NEW_DIM = 19
HIDDEN  = 256

LAYER_KEY = 'net.0.weight'  # nn.Sequential first linear layer

# Default local paths (vast.ai layout)
DEFAULT_SRC = 'checkpoints/btcusdt/nova_brain_best.pt'
DEFAULT_DST = 'checkpoints/btcusdt/nova_brain_19dim.pt'


# ---------------------------------------------------------------------------
# S3 helpers (only imported/used when paths start with "s3://")
# ---------------------------------------------------------------------------

def _get_s3():
    try:
        import boto3
        return boto3.client('s3')
    except ImportError:
        print("[ERROR] boto3 not installed. Run: pip install boto3")
        sys.exit(1)


def _parse_s3(path: str):
    """Parse 's3://bucket/key' → (bucket, key)."""
    assert path.startswith('s3://')
    parts = path[5:].split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def _load_checkpoint(path: str) -> dict:
    if path.startswith('s3://'):
        s3 = _get_s3()
        bucket, key = _parse_s3(path)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp = f.name
        s3.download_file(bucket, key, tmp)
        ckpt = torch.load(tmp, map_location='cpu', weights_only=False)
        os.unlink(tmp)
        print(f"[LOAD] s3://{bucket}/{key}")
        return ckpt
    else:
        path = Path(path)
        if not path.exists():
            print(f"[ERROR] Source not found: {path}")
            sys.exit(1)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"[LOAD] {path}")
        return ckpt


def _save_checkpoint(ckpt: dict, path: str):
    if path.startswith('s3://'):
        s3 = _get_s3()
        bucket, key = _parse_s3(path)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp = f.name
        torch.save(ckpt, tmp)
        s3.upload_file(tmp, bucket, key)
        os.unlink(tmp)
        print(f"[SAVE] s3://{bucket}/{key}")
    else:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        print(f"[SAVE] {out}")


def _backup_path(src: str) -> str:
    """Derive backup path: insert '_pre_19dim_backup' before .pt extension."""
    if src.startswith('s3://'):
        return src.replace('.pt', '_pre_19dim_backup.pt')
    p = Path(src)
    return str(p.with_stem(p.stem + '_pre_19dim_backup'))


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------

def extend_state_dict(sd: dict, label: str) -> dict:
    """Extend net.0.weight from [256,17] → [256,19] in a state_dict."""
    if LAYER_KEY not in sd:
        print(f"[ERROR] Key '{LAYER_KEY}' not found in {label} state_dict.")
        print(f"        Available keys: {list(sd.keys())}")
        sys.exit(1)

    old_weight = sd[LAYER_KEY]
    if old_weight.shape != (HIDDEN, OLD_DIM):
        print(f"[ERROR] {label}: expected shape ({HIDDEN}, {OLD_DIM}), got {tuple(old_weight.shape)}")
        sys.exit(1)

    # Two new columns — near-zero so existing policy survives day 0
    new_cols = torch.randn(HIDDEN, 2) * 0.001
    new_weight = torch.cat([old_weight, new_cols], dim=1)

    assert new_weight.shape == (HIDDEN, NEW_DIM), (
        f"Extension failed: expected ({HIDDEN}, {NEW_DIM}), got {tuple(new_weight.shape)}"
    )

    for i, col_idx in enumerate([17, 18]):
        col = new_cols[:, i]
        print(f"  {label} col[{col_idx}] — min={col.min():.6f}  max={col.max():.6f}  "
              f"mean={col.mean():.6f}  norm={col.norm():.6f}")

    sd = dict(sd)
    sd[LAYER_KEY] = new_weight
    return sd


def migrate(ckpt: dict) -> dict:
    """Migrate full checkpoint: extend actor, critic1, critic2 state_dicts."""
    print("\n[MIGRATE] Extending Actor:")
    ckpt['actor']   = extend_state_dict(ckpt['actor'],   'actor')
    print("[MIGRATE] Extending Critic1:")
    ckpt['critic1'] = extend_state_dict(ckpt['critic1'], 'critic1')
    print("[MIGRATE] Extending Critic2:")
    ckpt['critic2'] = extend_state_dict(ckpt['critic2'], 'critic2')

    # Extend target critics too if present
    for key in ('critic1_target', 'critic2_target'):
        if key in ckpt:
            print(f"[MIGRATE] Extending {key}:")
            ckpt[key] = extend_state_dict(ckpt[key], key)

    # Final shape assertions
    for net in ('actor', 'critic1', 'critic2'):
        shape = tuple(ckpt[net][LAYER_KEY].shape)
        assert shape == (HIDDEN, NEW_DIM), f"{net} shape wrong after migration: {shape}"
        print(f"  [OK] {net}: {shape}")

    return ckpt


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Migrate 17-dim SAC checkpoint to 19-dim')
    p.add_argument('--src',     default=DEFAULT_SRC,
                   help=f'Source checkpoint path or s3:// URI (default: {DEFAULT_SRC})')
    p.add_argument('--dst',     default=DEFAULT_DST,
                   help=f'Destination path or s3:// URI (default: {DEFAULT_DST})')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan and shapes, make no changes')
    args = p.parse_args()

    backup = _backup_path(args.src)

    print(f"[MIGRATE] Source  : {args.src}")
    print(f"[MIGRATE] Backup  : {backup}")
    print(f"[MIGRATE] Dest    : {args.dst}")
    print(f"[MIGRATE] Dry-run : {args.dry_run}")
    print(f"[MIGRATE] {OLD_DIM}-dim → {NEW_DIM}-dim  (adding vol_flow_30 [17], vol_flow_240 [18])")

    if args.dry_run:
        print("\n[DRY-RUN] Plan printed — run without --dry-run to execute.")
        return

    # Step 1: backup source before touching anything
    if args.src.startswith('s3://'):
        s3 = _get_s3()
        src_bucket, src_key = _parse_s3(args.src)
        bak_bucket, bak_key = _parse_s3(backup)
        s3.copy_object(Bucket=bak_bucket,
                       CopySource={'Bucket': src_bucket, 'Key': src_key},
                       Key=bak_key)
        print(f"[BACKUP] s3://{bak_bucket}/{bak_key}")
    else:
        src_p = Path(args.src)
        bak_p = Path(backup)
        shutil.copy2(src_p, bak_p)
        print(f"[BACKUP] {bak_p}")

    # Step 2: load
    ckpt = _load_checkpoint(args.src)
    print(f"[MIGRATE] Checkpoint keys: {list(ckpt.keys())}")
    for net in ('actor', 'critic1', 'critic2'):
        if net in ckpt:
            print(f"  {net} {LAYER_KEY}: {tuple(ckpt[net][LAYER_KEY].shape)}")

    # Step 3: migrate
    ckpt = migrate(ckpt)

    # Step 4: save
    _save_checkpoint(ckpt, args.dst)

    # Step 5: verify round-trip
    verify = _load_checkpoint(args.dst)
    for net in ('actor', 'critic1', 'critic2'):
        shape = tuple(verify[net][LAYER_KEY].shape)
        assert shape == (HIDDEN, NEW_DIM), f"Verify failed for {net}: {shape}"
    print(f"\n[MIGRATE] Verification passed — all networks are ({HIDDEN}, {NEW_DIM})")
    print(f"[MIGRATE] Done.")
    print(f"  Migrated checkpoint : {args.dst}")
    print(f"  Baseline backup     : {backup}")


if __name__ == '__main__':
    main()
