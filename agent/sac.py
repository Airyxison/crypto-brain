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
        self.batch_size   = cfg.get('batch_size',   1024)
        self.buffer_size  = cfg.get('buffer_size',   200_000)
        self.warmup_steps = cfg.get('warmup_steps',  1_000)

        # Auto-tuning temperature (Haarnoja et al. 2018).
        # Target entropy = 0.98 * log(n_actions) keeps the policy from collapsing to
        # near-deterministic HOLD — the failure mode seen in v8 with fixed alpha=0.03.
        # An initial alpha is still accepted for warm-starting; if auto_alpha=False
        # the old fixed-temperature mode is restored (useful for ablations).
        n_actions = 5
        self.auto_alpha = cfg.get('auto_alpha', True)
        self.target_entropy = -np.log(1.0 / n_actions) * cfg.get('entropy_factor', 0.98)
        init_alpha = cfg.get('alpha', 0.1)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                          requires_grad=True, device=self.device)
            alpha_lr = cfg.get('alpha_lr', self.lr * 0.1)  # slower than actor/critic
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha_value = self.log_alpha.exp().item()
        else:
            self.alpha_value = init_alpha
            self.log_alpha   = None
            self.alpha_opt   = None

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

        # Piecewise entropy decay (v12): after exploit_start_step, stop auto-tuning
        # and linearly decay log_alpha to exploit_floor. Captures log_alpha at the
        # transition point so resume-from-checkpoint works correctly.
        self.exploit_start_step  = cfg.get('exploit_start_step', None)   # None = disabled
        self.exploit_floor       = cfg.get('exploit_floor', -3.0)         # alpha ≈ 0.05
        self._log_alpha_at_transition = None

    @property
    def alpha(self) -> float:
        if self.auto_alpha:
            self.alpha_value = self.log_alpha.exp().item()
        return self.alpha_value

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if self.steps < self.warmup_steps:
            return np.random.randint(0, 5)
        return self.actor.act(state, deterministic=deterministic)

    def select_action_batch(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select actions for a batch of observations (vectorized env support).
        states: np.ndarray of shape (N, STATE_DIM)
        returns: np.ndarray of shape (N,) dtype int
        """
        if self.steps < self.warmup_steps:
            return np.random.randint(0, 5, size=len(states))
        state_t = torch.FloatTensor(states).to(self.device)  # (N, STATE_DIM)
        with torch.no_grad():
            probs, _ = self.actor(state_t)                   # (N, n_actions)
            if deterministic:
                actions = probs.argmax(dim=1)
            else:
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions.cpu().numpy()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def store_batch(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        rewards:     np.ndarray,
        next_states: np.ndarray,
        dones:       np.ndarray,
    ):
        """Store a batch of N transitions in one numpy op (vectorized env support)."""
        self.buffer.push_batch(states, actions, rewards, next_states, dones)

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

        # ---- Alpha (temperature) update ----
        # Phase 1 (steps < exploit_start_step): auto-tune alpha toward target_entropy.
        # Phase 2 (steps >= exploit_start_step): freeze auto-tuner, linearly decay
        #   log_alpha to exploit_floor so the agent exploits its learned policy.
        alpha_loss = None
        if self.auto_alpha:
            in_exploit_phase = (
                self.exploit_start_step is not None
                and self.steps >= self.exploit_start_step
            )
            if in_exploit_phase:
                # Capture transition value once
                if self._log_alpha_at_transition is None:
                    self._log_alpha_at_transition = self.log_alpha.data.item()
                # Linear decay from transition value → exploit_floor over remaining steps
                # steps beyond transition (unbounded — decays to floor then stays)
                steps_past = self.steps - self.exploit_start_step
                decay_per_step = (self.exploit_floor - self._log_alpha_at_transition) / max(self.exploit_start_step, 1)
                new_log_alpha = max(
                    self._log_alpha_at_transition + decay_per_step * steps_past,
                    self.exploit_floor
                )
                self.log_alpha.data.fill_(new_log_alpha)
                # alpha_loss stays None — optimizer not called in exploit phase
            else:
                alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                # Clamp log_alpha to prevent runaway alpha explosion (v9 lesson: alpha hit 2.67M).
                # min=-5.0 → alpha≥0.007 (never fully remove entropy bonus)
                # max=2.0  → alpha≤7.4  (Q-signal always visible to the actor)
                self.log_alpha.data.clamp_(min=-5.0, max=2.0)
            self.alpha_value = self.log_alpha.exp().item()

        # ---- Soft update target networks ----
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss':   actor_loss.item(),
            'alpha_loss':   alpha_loss.item() if alpha_loss is not None else 0.0,
            'alpha':        self.alpha,
            'entropy':      entropy.item(),
            'target_entropy': self.target_entropy,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def save(self, path: str):
        payload = {
            'actor':        self.actor.state_dict(),
            'critic1':      self.critic1.state_dict(),
            'critic2':      self.critic2.state_dict(),
            'alpha_value':  self.alpha_value,
            'steps':        self.steps,
            'best_sortino': self.best_sortino,
        }
        if self.auto_alpha and self.log_alpha is not None:
            payload['log_alpha'] = self.log_alpha.data
        torch.save(payload, path)
        print(f"[SAC] Saved checkpoint → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck['actor'])
        self.critic1.load_state_dict(ck['critic1'])
        self.critic2.load_state_dict(ck['critic2'])
        self.target1.load_state_dict(ck['critic1'])
        self.target2.load_state_dict(ck['critic2'])
        # Restore log_alpha only in auto_alpha mode — keeps alpha warm across restarts.
        # In fixed mode alpha is intentionally not restored (caller controls it).
        if self.auto_alpha and 'log_alpha' in ck:
            self.log_alpha.data = ck['log_alpha'].to(self.device)
            self.alpha_value = self.log_alpha.exp().item()
        self.steps = ck.get('steps', 0)
        self.best_sortino = ck.get('best_sortino', -np.inf)
        print(f"[SAC] Loaded checkpoint ← {path} (step {self.steps}, alpha {self.alpha_value:.4f}, best_sortino {self.best_sortino:.4f})")
