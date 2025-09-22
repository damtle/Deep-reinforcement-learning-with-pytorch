"""Deep Deterministic Policy Gradient agent."""
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.replay_buffer import ReplayBuffer
from rl_framework.networks.mlp import build_mlp
from rl_framework.types import Transition


class DeterministicPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.model = build_mlp(observation_dim, action_dim, hidden_sizes, activation=nn.ReLU)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        input_dim = observation_dim + action_dim
        self.model = build_mlp(input_dim, 1, hidden_sizes, activation=nn.ReLU)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation, action], dim=1)
        return self.model(x).squeeze(-1)


class DDPGAgent(BaseAgent):
    """Off-policy actor-critic agent for continuous control."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box) or not isinstance(action_space, gym.spaces.Box):
            raise TypeError("DDPG requires continuous (Box) observation and action spaces.")

        self.device = torch.device(device)
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        hidden_sizes = tuple(config.get("hidden_sizes", (400, 300)))

        self.actor = DeterministicPolicy(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target = DeterministicPolicy(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 1e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))

        buffer_size = int(config.get("buffer_size", 1_000_000))
        self.batch_size = int(config.get("batch_size", 128))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 5e-3))
        self.train_frequency = int(config.get("train_frequency", 1))
        self.gradient_steps = int(config.get("gradient_steps", 1))
        self.startup_steps = int(config.get("startup_steps", 1000))
        self.exploration_noise = float(config.get("exploration_noise", 0.1))

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            observation_shape=tuple(observation_space.shape),
            action_shape=tuple(action_space.shape),
            discrete_action=False,
        )

        self._steps = 0
    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def _select_deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        action = action.squeeze(0)
        return action.cpu().numpy()

    def select_action(self, observation, explore: bool = True) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        action = self._select_deterministic_action(observation)
        if explore:
            noise = np.random.normal(0.0, self.exploration_noise, size=action.shape)
            action = action + noise
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action.astype(np.float32)

    def observe(self, transition: Transition) -> bool:
        observation = np.asarray(transition.observation, dtype=np.float32)
        next_observation = np.asarray(transition.next_observation, dtype=np.float32)
        action = np.asarray(transition.action, dtype=np.float32)

        self.replay_buffer.add(observation, action, transition.reward, next_observation, transition.done)

        self._steps += 1
        if self._steps < self.startup_steps:
            return False
        return self._steps % self.train_frequency == 0

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        batch = self.replay_buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def update(self) -> Dict[str, Any]:
        info: Dict[str, float] = {}
        for _ in range(self.gradient_steps):
            if len(self.replay_buffer) < self.batch_size:
                return info
            batch = self._sample_batch()

            with torch.no_grad():
                next_actions = self.actor_target(batch["next_obs"])
                target_q = self.critic_target(batch["next_obs"], next_actions)
                target_value = batch["rewards"] + self.gamma * (1 - batch["dones"]) * target_q

            current_q = self.critic(batch["obs"], batch["actions"])
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(batch["obs"], self.actor(batch["obs"])).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            info = {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
            }
        return info
