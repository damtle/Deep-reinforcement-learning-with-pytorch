"""Rollout buffer for on-policy algorithms such as PPO."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch

class RolloutBuffer:
    """Fixed-size rollout storage supporting GAE computation."""

    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...], discrete_action: bool) -> None:
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.discrete_action = discrete_action

        self.reset()

    def reset(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.next_observations: list[np.ndarray] = []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rewards)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        if len(self.rewards) >= self.capacity:
            raise ValueError("Rollout buffer capacity exceeded. Call `reset` before storing more data.")
        observation_arr = np.asarray(observation, dtype=np.float32).reshape(self.observation_shape)
        next_observation_arr = np.asarray(next_observation, dtype=np.float32).reshape(self.observation_shape)
        self.observations.append(np.array(observation_arr, copy=True))
        self.next_observations.append(np.array(next_observation_arr, copy=True))
        if self.discrete_action:
            action_arr = np.asarray(action, dtype=np.int64).reshape(self.action_shape)
        else:
            action_arr = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        self.actions.append(np.array(action_arr, copy=True))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def compute_returns_and_advantages(
        self,
        value_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        gamma: float,
        gae_lambda: float,
    ) -> Dict[str, torch.Tensor]:
        if len(self.rewards) == 0:
            raise ValueError("Rollout buffer is empty. Nothing to compute.")

        observations = torch.as_tensor(np.asarray(self.observations), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.int64 if self.discrete_action else torch.float32, device=device)
        log_probs = torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device)
        values = torch.as_tensor(np.asarray(self.values), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.asarray(self.rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.asarray(self.dones), dtype=torch.float32, device=device)
        next_observations = torch.as_tensor(np.asarray(self.next_observations), dtype=torch.float32, device=device)

        with torch.no_grad():
            next_values = value_fn(next_observations).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(1, device=device)
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values[step] * mask - values[step]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[step] = last_advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }