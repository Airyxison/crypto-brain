"""
Training Script — Nova Brain POC
---------------------------------
Usage:
  python train.py --db ../crypto-engine/ticks.db --steps 50000

The script:
  1. Loads tick data from SQLite (written by the Rust engine)
  2. Splits 80/20 train/test
  3. Runs SAC training loop
  4. Saves checkpoint every N steps
  5. Runs backtest on test split at the end
"""

import argparse
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm

from environment.trading_env import TradingEnv
from agent.sac import SAC
from backtest.runner import load_ticks_from_db, run_backtest


def parse_args():
    p = argparse.ArgumentParser(description='Train Nova Brain SAC agent')
    p.add_argument('--db',       default='../crypto-engine/ticks.db', help='Path to SQLite tick DB')
    p.add_argument('--steps',    type=int, default=50_000,  help='Total training steps')
    p.add_argument('--save-dir', default='checkpoints',     help='Checkpoint directory')
    p.add_argument('--save-every', type=int, default=5_000, help='Save checkpoint every N steps')
    p.add_argument('--symbol',   default='BTCUSDT',         help='Asset symbol')
    p.add_argument('--resume',   default=None,              help='Resume from checkpoint path')
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load data
    print(f"[TRAIN] Loading ticks from {args.db}...")
    ticks = load_ticks_from_db(args.db, symbol=args.symbol)
    print(f"[TRAIN] {len(ticks)} ticks loaded.")
    if len(ticks) < 500:
        print("[TRAIN] Need at least 500 ticks — let the engine run a few more minutes.")
        return

    # 80/20 train/test split
    split     = int(len(ticks) * 0.8)
    train_ticks = ticks[:split]
    test_ticks  = ticks[split:]
    print(f"[TRAIN] {len(train_ticks)} train ticks | {len(test_ticks)} test ticks")

    # Initialize agent
    agent = SAC()
    if args.resume:
        agent.load(args.resume)

    # Training environment (resets randomly within train split)
    env = TradingEnv(train_ticks)
    obs, _ = env.reset()

    # Metrics tracking
    episode_rewards  = []
    episode_reward   = 0.0
    best_sortino     = -np.inf
    loss_log         = []
    action_counts    = [0] * 5
    ACTION_NAMES     = ['HOLD', 'BUY', 'ADJ_STOP', 'REALIZE', 'CANCEL']

    print(f"[TRAIN] Starting training for {args.steps} steps...")
    for step in tqdm(range(1, args.steps + 1), ncols=80):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_counts[action] += 1
        agent.store(obs, action, reward, next_obs, done)
        losses = agent.train_step()
        if losses:
            loss_log.append(losses)

        episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()

        # Periodic logging
        if step % 1000 == 0:
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            alpha   = loss_log[-1]['alpha']   if loss_log else 0
            entropy = loss_log[-1]['entropy'] if loss_log else 0
            total_a = max(sum(action_counts), 1)
            dist = '  '.join(f"{ACTION_NAMES[i]}:{action_counts[i]/total_a*100:.0f}%" for i in range(5))
            print(f"\n[Step {step:>6}] reward={avg_reward:+.4f}  α={alpha:.4f}  H={entropy:.4f}  episodes={len(episode_rewards)}")
            print(f"           actions → {dist}")

        # Checkpoint
        if step % args.save_every == 0:
            ck_path = save_dir / f'nova_brain_step{step}.pt'
            agent.save(str(ck_path))

            # Quick backtest on test split
            print("[TRAIN] Running validation backtest...")
            results = run_backtest(agent, test_ticks, verbose=True)
            sortino = results['sortino_ratio']

            if sortino > best_sortino:
                best_sortino = sortino
                best_path = save_dir / 'nova_brain_best.pt'
                agent.save(str(best_path))
                print(f"[TRAIN] New best Sortino: {sortino:.4f} → saved to {best_path}")

    # Final backtest
    print("\n[TRAIN] === FINAL BACKTEST ===")
    final_results = run_backtest(agent, test_ticks, verbose=True)

    # Save final
    agent.save(str(save_dir / 'nova_brain_final.pt'))
    print("[TRAIN] Done.")


if __name__ == '__main__':
    main()
