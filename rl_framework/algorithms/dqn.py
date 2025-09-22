"""Deep Q-Network implementation for the unified framework."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.replay_buffer import ReplayBuffer
from rl_framework.types import Transition


def _infer_observation_dim(space: gym.Space) -> Tuple[int, ...]:
    if isinstance(space, gym.spaces.Box):
        return tuple(space.shape)
    raise ValueError("DQN currently only supports Box observation spaces.")


def _infer_action_dim(space: gym.Space) -> Tuple[int, ...]:
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    raise ValueError("DQN currently only supports Discrete action spaces.")


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNAgent(BaseAgent):
    """A modern DQN agent with experience replay and target network."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        self.device = torch.device(device)

        observation_dim = int(np.prod(_infer_observation_dim(observation_space)))
        self.action_size = action_space.n

        hidden_sizes = tuple(config.get("hidden_sizes", (128, 128)))
        self.online_q = QNetwork(observation_dim, self.action_size, hidden_sizes).to(self.device)
        self.target_q = QNetwork(observation_dim, self.action_size, hidden_sizes).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())

        lr = config.get("lr", 1e-3)
        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=lr)

        buffer_size = config.get("buffer_size", 100_000)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.target_update_interval = config.get("target_update_interval", 1000)
        self.train_frequency = config.get("train_frequency", 1)
        self.min_replay_size = config.get("min_replay_size", 1000)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm")

        epsilon_cfg = config.get("epsilon", {})
        self.epsilon = epsilon_cfg.get("start", 1.0)
        self.min_epsilon = epsilon_cfg.get("end", 0.05)
        decay_steps = max(1, epsilon_cfg.get("decay_steps", 10_000))
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / decay_steps

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            observation_shape=_infer_observation_dim(observation_space),
            action_shape=_infer_action_dim(action_space),
            discrete_action=True,
        )

        self._steps = 0
        self._updates = 0

    def train_mode(self) -> None:
        self.online_q.train()
        self.target_q.train()

    def eval_mode(self) -> None:
        self.online_q.eval()
        self.target_q.eval()

    def select_action(self, observation, explore: bool = True) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(observation, device=self.device).view(1, -1)
        if explore and np.random.rand() < self.epsilon:
            action = self.action_space.sample()
            return int(action)
        with torch.no_grad():
            q_values = self.online_q(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)

    def observe(self, transition: Transition) -> bool:
        observation = np.asarray(transition.observation, dtype=np.float32)
        next_observation = np.asarray(transition.next_observation, dtype=np.float32)
        self.replay_buffer.add(observation, transition.action, transition.reward, next_observation, transition.done)

        self._steps += 1
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

        if len(self.replay_buffer) < self.min_replay_size:
            return False
        return self._steps % self.train_frequency == 0

    def update(self) -> Dict[str, Any]:
        losses = []
        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            obs = torch.as_tensor(batch.observations, device=self.device)
            actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).view(-1, 1)
            rewards = torch.as_tensor(batch.rewards, device=self.device).view(-1, 1)
            next_obs = torch.as_tensor(batch.next_observations, device=self.device)
            dones = torch.as_tensor(batch.dones, device=self.device).view(-1, 1)

            q_values = self.online_q(obs).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_q(next_obs).max(dim=1, keepdim=True)[0]
                targets = rewards + self.gamma * (1 - dones) * next_q_values

            loss = F.mse_loss(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.online_q.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(loss.item())

        self._updates += 1
        if self._updates % self.target_update_interval == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())

        return {"loss": float(np.mean(losses)), "epsilon": float(self.epsilon)}

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - persistence helper
        return {
            "online_q": self.online_q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self._steps,
            "updates": self._updates,
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # pragma: no cover - persistence helper
        self.online_q.load_state_dict(state_dict["online_q"])
        self.target_q.load_state_dict(state_dict["target_q"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epsilon = state_dict["epsilon"]
        self._steps = state_dict["steps"]
        self._updates = state_dict["updates"]
        self.replay_buffer.load_state_dict(state_dict["replay_buffer"])
