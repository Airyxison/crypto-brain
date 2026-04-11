"""
Queue Runner
------------
Reads run_queue.json and runs training jobs sequentially.
Safe to interrupt with Ctrl-C — updates status to 'failed' so you know where to restart.

Usage:
  python queue_runner.py                    # run all pending jobs
  python queue_runner.py --dry-run          # show what would run, don't train
  python queue_runner.py --id btc-v2-200k  # run a single job by id

The runner marks each job 'running' → 'completed' (or 'failed') in run_queue.json
so you always know the state even after a crash.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

QUEUE_FILE = Path(__file__).parent / 'run_queue.json'
TRAIN_CMD  = [sys.executable, 'train.py']


def load_queue() -> dict:
    with open(QUEUE_FILE) as f:
        return json.load(f)


def save_queue(data: dict):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def build_cmd(job: dict) -> list[str]:
    cmd = TRAIN_CMD + [
        '--symbol',     job['symbol'],
        '--steps',      str(job['steps']),
        '--save-dir',   'checkpoints',
        '--save-every', '10000',
    ]
    if job.get('alpha') is not None:
        cmd += ['--alpha', str(job['alpha'])]
    if job.get('resume'):
        cmd += ['--resume', job['resume']]
    return cmd


def run_job(job: dict, data: dict, dry_run: bool = False):
    cmd = build_cmd(job)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running job: {job['id']}")
    print(f"  symbol : {job['symbol']}")
    print(f"  steps  : {job['steps']}")
    print(f"  alpha  : {job.get('alpha', 'default')}")
    print(f"  resume : {job.get('resume') or 'none'}")
    print(f"  notes  : {job.get('notes', '')}")
    print(f"  cmd    : {' '.join(cmd)}")

    if dry_run:
        return

    # Mark running
    job['status']     = 'running'
    job['started_at'] = datetime.now(timezone.utc).isoformat()
    save_queue(data)

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode == 0:
            job['status']       = 'completed'
            job['completed_at'] = datetime.now(timezone.utc).isoformat()
            data['completed'].append(job)
            data['queue'].remove(job)
            print(f"\n[QUEUE] Job {job['id']} completed.")
        else:
            job['status'] = 'failed'
            print(f"\n[QUEUE] Job {job['id']} exited with code {result.returncode}.")
    except KeyboardInterrupt:
        job['status'] = 'interrupted'
        print(f"\n[QUEUE] Job {job['id']} interrupted by user.")
        raise
    finally:
        save_queue(data)


def main():
    parser = argparse.ArgumentParser(description='Sequential training queue runner')
    parser.add_argument('--dry-run', action='store_true', help='Show jobs without running')
    parser.add_argument('--id',      default=None,        help='Run a single job by id')
    args = parser.parse_args()

    data = load_queue()
    pending = [j for j in data['queue'] if j['status'] in ('pending', 'interrupted')]

    if args.id:
        pending = [j for j in pending if j['id'] == args.id]
        if not pending:
            print(f"[QUEUE] No pending job with id '{args.id}'")
            return

    if not pending:
        print("[QUEUE] No pending jobs.")
        return

    print(f"[QUEUE] {len(pending)} job(s) to run:")
    for j in pending:
        print(f"  {j['id']:25s}  {j['symbol']:10s}  {j['steps']:>8,} steps  alpha={j.get('alpha', 'default')}")

    try:
        for job in pending:
            run_job(job, data, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n[QUEUE] Runner stopped.")


if __name__ == '__main__':
    main()
