"""
Replay Buffer
-------------
Fixed-size circular buffer for experience replay.
Stores (state, action, reward, next_state, done) tuples.
Sampling is uniform random — prioritized replay is a future enhancement.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, state_dim: int = 13):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.ptr       = 0
        self.size      = 0

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,),           dtype=np.int64)
        self.rewards     = np.zeros((capacity,),           dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity,),           dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states':      torch.FloatTensor(self.states[idx]).to(device),
            'actions':     torch.LongTensor(self.actions[idx]).to(device),
            'rewards':     torch.FloatTensor(self.rewards[idx]).to(device),
            'next_states': torch.FloatTensor(self.next_states[idx]).to(device),
            'dones':       torch.FloatTensor(self.dones[idx]).to(device),
        }

    def __len__(self) -> int:
        return self.size
