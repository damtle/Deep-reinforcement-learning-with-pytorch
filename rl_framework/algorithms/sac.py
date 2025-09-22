"""Soft Actor-Critic agent."""
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.replay_buffer import ReplayBuffer
from rl_framework.networks.mlp import build_mlp
from rl_framework.types import Transition


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1 - tau)
        target_param.data.add_(tau * param.data)


class GaussianPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = observation_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = observation.view(observation.size(0), -1)
        if len(self.net) > 0:
            features = self.net(features)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = normal.rsample()
        action = torch.tanh(noise)
        log_prob = normal.log_prob(noise) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean)
        return action, log_prob, deterministic


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        input_dim = observation_dim + action_dim
        self.net = build_mlp(input_dim, 1, hidden_sizes, activation=nn.ReLU)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        observation = observation.view(observation.size(0), -1)
        action = action.view(action.size(0), -1)
        x = torch.cat([observation, action], dim=1)
        return self.net(x).squeeze(-1)


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for continuous control."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box) or not isinstance(action_space, gym.spaces.Box):
            raise TypeError("SAC requires continuous (Box) observation and action spaces.")

        self.device = torch.device(device)
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        hidden_sizes = tuple(config.get("hidden_sizes", (256, 256)))

        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_critic1 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("actor_lr", 3e-4))
        critic_lr = config.get("critic_lr", 3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.automatic_entropy_tuning = bool(config.get("automatic_entropy_tuning", True))
        if self.automatic_entropy_tuning:
            self.target_entropy = float(config.get("target_entropy", -action_dim))
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 3e-4))
        else:
            self.alpha = float(config.get("alpha", 0.2))
            self.log_alpha = None
            self.alpha_optimizer = None

        buffer_size = int(config.get("buffer_size", 1_000_000))
        self.batch_size = int(config.get("batch_size", 256))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 5e-3))
        self.gradient_steps = int(config.get("gradient_steps", 1))
        self.train_frequency = int(config.get("train_frequency", 1))
        self.startup_steps = int(config.get("startup_steps", 1000))

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            observation_shape=tuple(observation_space.shape),
            action_shape=tuple(action_space.shape),
            discrete_action=False,
        )

        action_scale = (action_space.high - action_space.low) / 2.0
        action_bias = (action_space.high + action_space.low) / 2.0
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor(action_bias, dtype=torch.float32, device=self.device)

        self._steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(self._alpha_static, device=self.device)
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha_static = value

    def train_mode(self) -> None:
        self.policy.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self) -> None:
        self.policy.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _select_action(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action, _, deterministic = self.policy.sample(obs_tensor)
            if not explore:
                action = deterministic
        action = action.squeeze(0)
        scaled = action * self.action_scale + self.action_bias
        return scaled.cpu().numpy()

    def select_action(self, observation, explore: bool = True) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        return self._select_action(observation, explore)

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
        return {"obs": obs, "actions": actions, "rewards": rewards, "next_obs": next_obs, "dones": dones}

    def update(self) -> Dict[str, Any]:
        metrics: Dict[str, float] = {}
        for _ in range(self.gradient_steps):
            if len(self.replay_buffer) < self.batch_size:
                return metrics
            batch = self._sample_batch()

            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy.sample(batch["next_obs"])
                next_action_scaled = next_action * self.action_scale + self.action_bias
                target_q1 = self.target_critic1(batch["next_obs"], next_action_scaled)
                target_q2 = self.target_critic2(batch["next_obs"], next_action_scaled)
                min_target = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.squeeze(-1)
                target_value = batch["rewards"] + self.gamma * (1 - batch["dones"]) * min_target

            current_q1 = self.critic1(batch["obs"], batch["actions"])
            current_q2 = self.critic2(batch["obs"], batch["actions"])
            critic1_loss = F.mse_loss(current_q1, target_value)
            critic2_loss = F.mse_loss(current_q2, target_value)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            new_action, log_prob, _ = self.policy.sample(batch["obs"])
            new_action_scaled = new_action * self.action_scale + self.action_bias
            q1_new = self.critic1(batch["obs"], new_action_scaled)
            q2_new = self.critic2(batch["obs"], new_action_scaled)
            actor_loss = (self.alpha * log_prob.squeeze(-1) - torch.min(q1_new, q2_new)).mean()

            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            alpha_loss_value = 0.0
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob.detach().squeeze(-1) + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.item())
            else:
                alpha_loss_value = 0.0

            _soft_update(self.critic1, self.target_critic1, self.tau)
            _soft_update(self.critic2, self.target_critic2, self.tau)

            metrics = {
                "critic1_loss": float(critic1_loss.item()),
                "critic2_loss": float(critic2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": alpha_loss_value,
                "alpha": float(self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha),
            }
        return metrics
