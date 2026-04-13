"""
Neural Networks — SAC (Discrete Action Space)
---------------------------------------------
Actor:   state (16) → action logits (5) → softmax policy
Critic:  state (16) → Q values for all 5 actions
Two critic networks (Q1, Q2) for clipped double-Q to reduce overestimation.

Architecture kept intentionally small for POC — scale up after proving the loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM  = 16  # 13 base + 3 macro regime features (momentum_1d/7d/30d)
ACTION_DIM = 5
HIDDEN     = 256  # scaled up for GPU utilization (T4 was underloaded at 128)


class Actor(nn.Module):
    """
    Outputs a probability distribution over discrete actions.
    Uses log_softmax for numerical stability.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, log_action_probs)"""
        logits    = self.net(state)
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (action, log_prob_of_action).
        Used during training for policy gradient.
        """
        probs, log_probs = self.forward(state)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Inference-time action selection."""
        import numpy as np
        device = next(self.parameters()).device
        t = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, _ = self.forward(t)
        if deterministic:
            return int(probs.argmax(dim=-1).item())
        return int(torch.multinomial(probs, 1).item())


class Critic(nn.Module):
    """
    Q-network for discrete actions.
    Input: state only (not state+action, since action space is small).
    Output: Q(s, a) for all actions simultaneously.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q values for all actions. Shape: (batch, action_dim)"""
        return self.net(state)

