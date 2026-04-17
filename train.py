"""
Training Script — Nova Brain POC
---------------------------------
Usage:
  python train.py --db /root/crypto-engine/ticks.db --steps 50000

The script:
  1. Loads tick data from SQLite (written by the Rust engine)
  2. Splits 80/20 train/test
  3. Runs SAC training loop
  4. Saves checkpoint every N steps
  5. Runs backtest on test split at the end

W&B dashboard:
  Set WANDB_API_KEY env var to enable. Each symbol runs as a separate W&B run
  so you can compare BTC/ETH/SOL/ADA side-by-side. Skipped silently if unset.
"""

import argparse
import os
from pathlib import Path

# Load .env if present — keeps secrets out of shell history and global profile
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _, _v = _line.partition('=')
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
from tqdm import tqdm

from environment.trading_env import TradingEnv
from agent.sac import SAC
from backtest.runner import load_ticks_from_db, run_backtest


# ---------------------------------------------------------------------------
# W&B — opt-in, no-op if API key not set
# ---------------------------------------------------------------------------

def wandb_init(args, agent):
    """Initialize W&B run. Returns run object or None."""
    if not os.environ.get('WANDB_API_KEY'):
        print("[W&B] WANDB_API_KEY not set — skipping dashboard.")
        return None
    try:
        import wandb
        auto_tag = "auto" if not args.no_auto_alpha else f"fixed{args.alpha}"
        run = wandb.init(
            project='nova-brain',
            name=f"{args.symbol}-v{args.steps//1000}k-{auto_tag}",
            config={
                'symbol':        args.symbol,
                'steps':         args.steps,
                'alpha_init':    args.alpha,
                'auto_alpha':    not args.no_auto_alpha,
                'episode_steps': args.episode_steps,
                'save_every':    args.save_every,
            },
            reinit=True,
        )
        print(f"[W&B] Dashboard → {run.url}")
        return run
    except Exception as e:
        print(f"[W&B] Init failed ({e}) — continuing without dashboard.")
        return None


def wandb_log(run, payload: dict):
    if run is None:
        return
    try:
        import wandb
        wandb.log(payload)
    except Exception:
        pass


def wandb_finish(run):
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# S3 checkpoint sync
# ---------------------------------------------------------------------------

def s3_upload(local_path: str, s3_key: str, bucket: str = 'nova-trader-data-249899228939-us-east-1-an'):
    """Upload a file to S3. Silently skips if boto3/credentials unavailable."""
    try:
        import boto3
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket, s3_key)
        print(f"[S3] Uploaded → s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[S3] Upload skipped ({e})")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train Nova Brain SAC agent')
    p.add_argument('--db',         default='/root/crypto-engine/ticks.db', help='Path to SQLite tick DB')
    p.add_argument('--steps',      type=int, default=200_000, help='Total training steps')
    p.add_argument('--save-dir',   default='checkpoints',     help='Checkpoint directory')
    p.add_argument('--save-every', type=int, default=10_000,  help='Save checkpoint every N steps')
    p.add_argument('--symbol',     default='BTCUSDT',         help='Asset symbol')
    p.add_argument('--resume',     default=None,              help='Resume from checkpoint path')
    p.add_argument('--alpha',         type=float, default=None,  help='Initial alpha / fixed alpha if --no-auto-alpha')
    p.add_argument('--no-auto-alpha', action='store_true',        help='Disable auto-alpha tuning (fixed temperature)')
    p.add_argument('--episode-steps', type=int,   default=1000,  help='Max steps per training episode (default 1000)')
    p.add_argument('--num-envs',      type=int,   default=1,     help='Number of parallel envs per symbol (vectorized training, v11+). Default 1 = serial loop.')
    p.add_argument('--buffer-size',   type=int,   default=None,  help='Replay buffer capacity. Defaults to 200k (serial) or 200k×num_envs (vectorized).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    save_dir = Path(args.save_dir) / args.symbol.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[TRAIN] Loading ticks from {args.db}...")
    ticks = load_ticks_from_db(args.db, symbol=args.symbol)
    print(f"[TRAIN] {len(ticks)} ticks loaded.")
    if len(ticks) < 500:
        print("[TRAIN] Need at least 500 ticks — let the engine run a few more minutes.")
        return

    # 80/20 train/test split
    split       = int(len(ticks) * 0.8)
    train_ticks = ticks[:split]
    test_ticks  = ticks[split:]
    print(f"[TRAIN] {len(train_ticks)} train ticks | {len(test_ticks)} test ticks")

    # Initialize agent
    sac_cfg = {}
    if args.alpha is not None:
        sac_cfg['alpha'] = args.alpha
    if args.no_auto_alpha:
        sac_cfg['auto_alpha'] = False
    # Scale buffer with num_envs to preserve experience diversity window.
    # Default: 200k serial, or 200k × num_envs vectorized (e.g. 3.2M for 16 envs).
    buf_size = args.buffer_size or (200_000 * max(args.num_envs, 1))
    sac_cfg['buffer_size'] = buf_size
    print(f"[TRAIN] Replay buffer: {buf_size:,} transitions")
    agent = SAC(config=sac_cfg)
    if args.resume:
        agent.load(args.resume)
    print(f"[TRAIN] Auto-alpha: {'ON (target_entropy={:.3f})'.format(agent.target_entropy) if agent.auto_alpha else 'OFF'}")

    # W&B run — no-op if WANDB_API_KEY not set
    wb = wandb_init(args, agent)

    # Training environment setup — serial (default) or vectorized (--num-envs N)
    if args.num_envs > 1:
        from gymnasium.vector import AsyncVectorEnv
        ep_steps = args.episode_steps
        vec_env = AsyncVectorEnv([
            (lambda: TradingEnv(train_ticks, {'max_episode_steps': ep_steps}))
            for _ in range(args.num_envs)
        ])
        obs_batch, _ = vec_env.reset()
        ep_rewards_vec = np.zeros(args.num_envs)
        grad_updates   = 2   # 2 updates/step: N envs push N transitions each step,
                             # keeping the experience/gradient ratio balanced (Nova v11 spec)
        print(f"[TRAIN] Vectorized mode: {args.num_envs} envs  |  {grad_updates} grad updates/step")
    else:
        env = TradingEnv(train_ticks, {'max_episode_steps': args.episode_steps})
        obs, _ = env.reset()
        episode_reward = 0.0
        grad_updates   = 4   # 4 updates/step in serial mode (unchanged from v10)

    # Metrics tracking (shared by both modes)
    episode_rewards = []
    best_sortino    = agent.best_sortino
    print(f"[TRAIN] best_sortino initialized to {best_sortino:.4f}")
    loss_log      = []
    ACTION_NAMES  = ['HOLD', 'BUY', 'ADJ_STOP', 'REALIZE', 'CANCEL']
    action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    print(f"[TRAIN] Starting training for {args.steps} steps...")
    for step in tqdm(range(1, args.steps + 1), ncols=80):

        if args.num_envs > 1:
            # ---- Vectorized step ------------------------------------------------
            # gymnasium 1.2.3: infos is a dict of arrays, e.g.
            #   infos['action_taken'] -> np.ndarray of shape (num_envs,)
            actions = agent.select_action_batch(obs_batch)
            next_obs_batch, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = terminated | truncated

            action_taken_arr = np.asarray(infos.get('action_taken', actions))
            # Batch insert — one numpy op instead of N Python iterations
            np.add.at(action_counts, action_taken_arr, 1)
            agent.store_batch(obs_batch, action_taken_arr, rewards, next_obs_batch, dones)

            ep_rewards_vec += rewards
            for i in np.where(dones)[0]:
                episode_rewards.append(float(ep_rewards_vec[i]))
                ep_rewards_vec[i] = 0.0

            obs_batch = next_obs_batch

        else:
            # ---- Serial step (unchanged from v10) --------------------------------
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store the shaped action (what the env actually executed), not the raw action.
            # This ensures the SAC never gets credit for invalid-action no-ops.
            action_taken = info.get('action_taken', action)
            action_counts[action_taken] += 1
            agent.store(obs, action_taken, reward, next_obs, done)

            episode_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

        # Gradient updates (4 for serial, 2 for vectorized — shared)
        losses = None
        for _ in range(grad_updates):
            result = agent.train_step()
            if result:
                losses = result
        if losses:
            loss_log.append(losses)

        # Periodic logging — every 1k steps
        if step % 1000 == 0:
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            alpha      = loss_log[-1]['alpha']        if loss_log else 0
            entropy    = loss_log[-1]['entropy']      if loss_log else 0
            tgt_ent    = loss_log[-1].get('target_entropy', 0) if loss_log else 0
            c1_loss    = loss_log[-1]['critic1_loss'] if loss_log else 0
            actor_loss = loss_log[-1]['actor_loss']   if loss_log else 0
            alpha_loss = loss_log[-1].get('alpha_loss', 0) if loss_log else 0
            total_a    = max(sum(action_counts), 1)
            dist = '  '.join(f"{ACTION_NAMES[i]}:{action_counts[i]/total_a*100:.0f}%" for i in range(5))

            print(f"\n[Step {step:>6}] reward={avg_reward:+.4f}  α={alpha:.4f}  H={entropy:.4f}  H_tgt={tgt_ent:.4f}  episodes={len(episode_rewards)}")
            print(f"           actions → {dist}")

            wandb_log(wb, {
                'step':               step,
                'reward/avg_10ep':    avg_reward,
                'train/entropy':      entropy,
                'train/entropy_target': tgt_ent,
                'train/alpha':        alpha,
                'train/alpha_loss':   alpha_loss,
                'train/critic_loss':  c1_loss,
                'train/actor_loss':   actor_loss,
                'actions/hold':       action_counts[0] / total_a,
                'actions/buy':        action_counts[1] / total_a,
                'actions/adj_stop':   action_counts[2] / total_a,
                'actions/realize':    action_counts[3] / total_a,
                'actions/cancel':     action_counts[4] / total_a,
                'episodes':           len(episode_rewards),
            })

        # Checkpoint + validation backtest
        if step % args.save_every == 0:
            ck_path = save_dir / f'nova_brain_step{step}.pt'
            agent.save(str(ck_path))
            s3_upload(str(ck_path), f'checkpoints/{args.symbol.lower()}/nova_brain_step{step}.pt')

            print("[TRAIN] Running validation backtest...")
            results   = run_backtest(agent, test_ticks, verbose=True)
            sortino   = results['sortino_ratio']
            total_a   = max(sum(action_counts), 1)

            wandb_log(wb, {
                'step':              step,
                'val/sortino':       sortino,
                'val/total_return':  results['total_return_pct'],
                'val/win_rate':      results['win_rate_pct'],
                'val/total_trades':  results['total_trades'],
                'val/avg_hold_bars': results['avg_hold_bars'],
                'val/max_drawdown':  results['max_drawdown_pct'],
                'val/final_value':   results['final_value'],
            })

            total_trades = results.get('total_trades', 0)
            if sortino > best_sortino and total_trades > 0:
                best_sortino       = sortino
                agent.best_sortino = best_sortino
                best_path = save_dir / 'nova_brain_best.pt'
                agent.save(str(best_path))
                s3_upload(str(best_path), f'checkpoints/{args.symbol.lower()}/nova_brain_best.pt')
                print(f"[TRAIN] New best Sortino: {sortino:.4f} → saved to {best_path}")
                wandb_log(wb, {'val/best_sortino': best_sortino, 'step': step})

    # Final backtest
    print("\n[TRAIN] === FINAL BACKTEST ===")
    final_results = run_backtest(agent, test_ticks, verbose=True)

    wandb_log(wb, {
        'step':                   args.steps,
        'final/sortino':          final_results['sortino_ratio'],
        'final/total_return':     final_results['total_return_pct'],
        'final/win_rate':         final_results['win_rate_pct'],
        'final/total_trades':     final_results['total_trades'],
        'final/max_drawdown':     final_results['max_drawdown_pct'],
    })

    # Save final
    final_path = save_dir / 'nova_brain_final.pt'
    agent.save(str(final_path))
    s3_upload(str(final_path), f'checkpoints/{args.symbol.lower()}/nova_brain_final.pt')

    wandb_finish(wb)
    print("[TRAIN] Done.")


if __name__ == '__main__':
    main()
