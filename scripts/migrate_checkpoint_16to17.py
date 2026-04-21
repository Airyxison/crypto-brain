"""
migrate_checkpoint_16to17.py
-----------------------------
One-off migration script: extends a 16-dim SAC checkpoint to 17-dim.

Extends Actor and Critic first linear layers (net.0.weight) from [256,16]→[256,17].
New column initialized near-zero so the existing policy is preserved at step 0.

Usage:
    python scripts/migrate_checkpoint_16to17.py [--dry-run]
    python scripts/migrate_checkpoint_16to17.py --src s3://bucket/key --dst s3://bucket/key

Steps:
    1. Copy-tag source as nova_brain_v12.1_best_sortino0.1076.pt (protect baseline)
    2. Load source checkpoint from S3
    3. Extend net.0.weight for Actor and Critic
    4. Assert shapes are correct
    5. Save migrated checkpoint to S3
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch


S3_BUCKET    = 'nova-trader-data-249899228939-us-east-1-an'
SRC_KEY      = 'checkpoints/btcusdt/nova_brain_best.pt'
TAG_KEY      = 'checkpoints/btcusdt/nova_brain_v12.1_best_sortino0.1076.pt'
DST_KEY      = 'checkpoints/btcusdt/nova_brain_v12.1_migrated_17dim.pt'

OLD_DIM = 16
NEW_DIM = 17
HIDDEN  = 256

# Keys in the checkpoint's actor/critic state_dicts that need extension
LAYER_KEY = 'net.0.weight'  # nn.Sequential first linear layer — NOT fc1.weight


def get_s3():
    try:
        import boto3
        return boto3.client('s3')
    except ImportError:
        print("[ERROR] boto3 not installed. Run: pip install boto3")
        sys.exit(1)


def s3_copy_tag(s3, bucket: str, src_key: str, dst_key: str, dry_run: bool):
    """Copy src → dst within same bucket (preserves baseline before any modification)."""
    if dry_run:
        print(f"[DRY-RUN] Would copy s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}")
        return
    try:
        s3.copy_object(
            Bucket=bucket,
            CopySource={'Bucket': bucket, 'Key': src_key},
            Key=dst_key,
        )
        print(f"[S3] Copy-tagged → s3://{bucket}/{dst_key}")
    except Exception as e:
        print(f"[ERROR] copy_tag failed: {e}")
        sys.exit(1)


def s3_download(s3, bucket: str, key: str, local_path: str, dry_run: bool):
    if dry_run:
        print(f"[DRY-RUN] Would download s3://{bucket}/{key} → {local_path}")
        return
    try:
        s3.download_file(bucket, key, local_path)
        print(f"[S3] Downloaded s3://{bucket}/{key}")
    except Exception as e:
        print(f"[ERROR] download failed: {e}")
        sys.exit(1)


def s3_upload(s3, local_path: str, bucket: str, key: str, dry_run: bool):
    if dry_run:
        print(f"[DRY-RUN] Would upload {local_path} → s3://{bucket}/{key}")
        return
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"[S3] Uploaded → s3://{bucket}/{key}")
    except Exception as e:
        print(f"[ERROR] upload failed: {e}")
        sys.exit(1)


def extend_state_dict(sd: dict, dry_run: bool) -> dict:
    """Extend net.0.weight from [256,16] → [256,17] in a state_dict."""
    if LAYER_KEY not in sd:
        keys = list(sd.keys())
        print(f"[ERROR] Key '{LAYER_KEY}' not found in state_dict. Available keys: {keys}")
        sys.exit(1)

    old_weight = sd[LAYER_KEY]
    assert old_weight.shape == (HIDDEN, OLD_DIM), (
        f"Expected shape ({HIDDEN}, {OLD_DIM}), got {tuple(old_weight.shape)}"
    )

    # Near-zero init: agent policy unchanged at step 0; gradients grow column [16] gradually
    new_col = torch.zeros(HIDDEN, 1) * 0.01 + torch.randn(HIDDEN, 1) * 0.001
    new_weight = torch.cat([old_weight, new_col], dim=1)

    assert new_weight.shape == (HIDDEN, NEW_DIM), (
        f"Extension failed: expected ({HIDDEN}, {NEW_DIM}), got {tuple(new_weight.shape)}"
    )

    col_stats = new_col.squeeze()
    print(f"  new column stats — min={col_stats.min():.6f}  max={col_stats.max():.6f}  mean={col_stats.mean():.6f}  norm={col_stats.norm():.6f}")

    sd = dict(sd)  # shallow copy to avoid modifying original
    sd[LAYER_KEY] = new_weight
    return sd


def migrate(ckpt: dict) -> dict:
    """Migrate full checkpoint dict: extend actor, critic1, and critic2 state_dicts."""
    print("\n[MIGRATE] Actor net.0.weight:")
    ckpt['actor']   = extend_state_dict(ckpt['actor'],   dry_run=False)
    print("[MIGRATE] Critic1 net.0.weight:")
    ckpt['critic1'] = extend_state_dict(ckpt['critic1'], dry_run=False)
    print("[MIGRATE] Critic2 net.0.weight:")
    ckpt['critic2'] = extend_state_dict(ckpt['critic2'], dry_run=False)

    # Final shape assertions
    a_shape  = tuple(ckpt['actor'][LAYER_KEY].shape)
    c1_shape = tuple(ckpt['critic1'][LAYER_KEY].shape)
    c2_shape = tuple(ckpt['critic2'][LAYER_KEY].shape)
    assert a_shape  == (HIDDEN, NEW_DIM), f"Actor shape wrong: {a_shape}"
    assert c1_shape == (HIDDEN, NEW_DIM), f"Critic1 shape wrong: {c1_shape}"
    assert c2_shape == (HIDDEN, NEW_DIM), f"Critic2 shape wrong: {c2_shape}"
    print(f"\n[MIGRATE] Assertions passed — actor {a_shape}, critic1 {c1_shape}, critic2 {c2_shape}")

    return ckpt


def main():
    p = argparse.ArgumentParser(description='Migrate 16-dim SAC checkpoint to 17-dim')
    p.add_argument('--src',     default=None,    help=f'S3 key for source checkpoint (default: {SRC_KEY})')
    p.add_argument('--dst',     default=None,    help=f'S3 key for output checkpoint (default: {DST_KEY})')
    p.add_argument('--bucket',  default=S3_BUCKET, help='S3 bucket')
    p.add_argument('--dry-run', action='store_true', help='Print shapes and plan, no saves')
    args = p.parse_args()

    src_key = args.src or SRC_KEY
    dst_key = args.dst or DST_KEY
    bucket  = args.bucket

    print(f"[MIGRATE] Source : s3://{bucket}/{src_key}")
    print(f"[MIGRATE] Dest   : s3://{bucket}/{dst_key}")
    print(f"[MIGRATE] Tag    : s3://{bucket}/{TAG_KEY}")
    print(f"[MIGRATE] Dry-run: {args.dry_run}")

    s3 = get_s3()

    # Step 1: copy-tag BEFORE touching anything (protect baseline)
    s3_copy_tag(s3, bucket, src_key, TAG_KEY, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN] Migration plan complete. Run without --dry-run to execute.")
        return

    # Step 2: download source
    with tempfile.TemporaryDirectory() as tmpdir:
        local_src = os.path.join(tmpdir, 'source.pt')
        local_dst = os.path.join(tmpdir, 'migrated.pt')

        s3_download(s3, bucket, src_key, local_src, dry_run=False)

        # Step 3: load and inspect
        ckpt = torch.load(local_src, map_location='cpu', weights_only=False)
        print(f"\n[MIGRATE] Checkpoint keys: {list(ckpt.keys())}")
        print(f"[MIGRATE] actor   net.0.weight shape (before): {tuple(ckpt['actor'][LAYER_KEY].shape)}")
        print(f"[MIGRATE] critic1 net.0.weight shape (before): {tuple(ckpt['critic1'][LAYER_KEY].shape)}")
        print(f"[MIGRATE] critic2 net.0.weight shape (before): {tuple(ckpt['critic2'][LAYER_KEY].shape)}")

        # Step 4: migrate
        ckpt = migrate(ckpt)

        # Step 5: save to temp file, then upload
        torch.save(ckpt, local_dst)
        print(f"[MIGRATE] Saved migrated checkpoint locally: {local_dst}")

        # Verify the saved file loads correctly
        verify = torch.load(local_dst, map_location='cpu', weights_only=False)
        assert tuple(verify['actor'][LAYER_KEY].shape)  == (HIDDEN, NEW_DIM), "Verify actor failed"
        assert tuple(verify['critic'][LAYER_KEY].shape) == (HIDDEN, NEW_DIM), "Verify critic failed"
        print("[MIGRATE] Verification passed — saved file loads and shapes are correct.")

        # Step 6: upload migrated checkpoint
        s3_upload(s3, local_dst, bucket, dst_key, dry_run=False)

    print(f"\n[MIGRATE] Done. Migrated checkpoint at s3://{bucket}/{dst_key}")
    print(f"[MIGRATE] Baseline preserved at s3://{bucket}/{TAG_KEY}")


if __name__ == '__main__':
    main()
