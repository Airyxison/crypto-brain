"""
Soft Actor-Critic (Discrete Action Space)
------------------------------------------
SAC maximizes a trade-off between expected return AND policy entropy:

    π* = argmax E[Σ r_t + α · H(π(·|s_t))]

The entropy term α · H(π) is the key property that prevents rigidity.
The agent is rewarded for staying uncertain (exploratory) until it has
a strong reason to commit — which is exactly the behavior we want for
opportunistic pattern recognition without getting trapped.

References:
  - Haarnoja et al., "Soft Actor-Critic" (2018)
  - Christodoulou, "Discrete SAC" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer
from agent.networks import STATE_DIM


class SAC:
    def __init__(self, config: dict | None = None):
        cfg = config or {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SAC] Using device: {self.device}")

        # Hyperparameters
        self.gamma        = cfg.get('gamma',        0.99)
        self.tau          = cfg.get('tau',           0.005)   # soft update rate
        self.lr           = cfg.get('lr',            3e-4)
        self.batch_size   = cfg.get('batch_size',    256)
        self.buffer_size  = cfg.get('buffer_size',   200_000)
        self.warmup_steps = cfg.get('warmup_steps',  1_000)

        # Fixed temperature for POC — auto-tuning added back once loop is stable
        # 0.1 pushes the policy to exploit Q-values more strongly; was 0.2 but
        # entropy barely moved after 50k steps so the exploration bonus was too dominant
        self.alpha_value = cfg.get('alpha', 0.1)
        self.log_alpha   = None  # not used in fixed mode
        self.target_entropy = None

        # Networks
        self.actor    = Actor().to(self.device)
        self.critic1  = Critic().to(self.device)
        self.critic2  = Critic().to(self.device)
        self.target1  = Critic().to(self.device)
        self.target2  = Critic().to(self.device)

        # Initialize targets as hard copies
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=self.lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=self.buffer_size, state_dim=STATE_DIM)

        self.steps = 0
        self.best_sortino = -np.inf

    @property
    def alpha(self) -> float:
        return self.alpha_value

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if self.steps < self.warmup_steps:
            return np.random.randint(0, 5)
        return self.actor.act(state, deterministic=deterministic)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> dict[str, float] | None:
        if len(self.buffer) < self.batch_size:
            return None

        self.steps += 1
        batch = self.buffer.sample(self.batch_size, self.device)
        s  = batch['states']
        a  = batch['actions']
        r  = batch['rewards']
        s_ = batch['next_states']
        d  = batch['dones']

        # ---- Critic update ----
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(s_)
            q1_next = self.target1(s_)
            q2_next = self.target2(s_)
            min_q_next = torch.min(q1_next, q2_next)

            # Soft Bellman: V(s') = Σ_a π(a|s') [Q(s',a) - α log π(a|s')]
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            q_target = r + (1.0 - d) * self.gamma * v_next.detach()

        q1 = self.critic1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(s).gather(1, a.unsqueeze(1)).squeeze(1)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        # ---- Actor update ----
        probs, log_probs = self.actor(s)
        with torch.no_grad():
            q1_s = self.critic1(s)
            q2_s = self.critic2(s)
            min_q = torch.min(q1_s, q2_s)

        # Maximize E[Q] + entropy
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        entropy = -(probs * log_probs).sum(dim=1).mean()

        # ---- Soft update target networks ----
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss':   actor_loss.item(),
            'alpha':        self.alpha,
            'entropy':      entropy.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def save(self, path: str):
        torch.save({
            'actor':        self.actor.state_dict(),
            'critic1':      self.critic1.state_dict(),
            'critic2':      self.critic2.state_dict(),
            'alpha_value':  self.alpha_value,
            'steps':        self.steps,
            'best_sortino': self.best_sortino,
        }, path)
        print(f"[SAC] Saved checkpoint → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck['actor'])
        self.critic1.load_state_dict(ck['critic1'])
        self.critic2.load_state_dict(ck['critic2'])
        self.target1.load_state_dict(ck['critic1'])
        self.target2.load_state_dict(ck['critic2'])
        # alpha is a hyperparameter — do not restore from checkpoint so
        # the caller can change it between runs without being overridden.
        self.steps = ck.get('steps', 0)
        self.best_sortino = ck.get('best_sortino', -np.inf)
        print(f"[SAC] Loaded checkpoint ← {path} (step {self.steps}, best_sortino {self.best_sortino:.4f})")
