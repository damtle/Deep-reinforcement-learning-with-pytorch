"""Replay buffer implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ReplaySample:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """A simple uniform replay buffer."""

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        discrete_action: bool,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.discrete_action = discrete_action

        self.observations = np.zeros((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *observation_shape), dtype=np.float32)
        action_dtype = np.int64 if discrete_action else np.float32
        self.actions = np.zeros((self.capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self._index = 0
        self._size = 0

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, next_observation: np.ndarray, done: bool) -> None:
        np.copyto(self.observations[self._index], observation)
        np.copyto(self.next_observations[self._index], next_observation)
        if self.discrete_action:
            self.actions[self._index] = action
        else:
            np.copyto(self.actions[self._index], action)
        self.rewards[self._index] = reward
        self.dones[self._index] = float(done)

        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplaySample:
        if self._size < batch_size:
            raise ValueError("Not enough samples in the replay buffer to sample the requested batch size.")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return ReplaySample(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "index": np.array([self._index], dtype=np.int64),
            "size": np.array([self._size], dtype=np.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        for key in ("observations", "actions", "rewards", "next_observations", "dones"):
            np.copyto(getattr(self, key), state_dict[key])
        self._index = int(state_dict["index"][0])
        self._size = int(state_dict["size"][0])
