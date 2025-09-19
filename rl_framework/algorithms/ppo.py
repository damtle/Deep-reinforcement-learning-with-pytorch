"""Proximal Policy Optimization implementation for the unified framework."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.rollout_buffer import RolloutBuffer
from rl_framework.types import Transition


def _infer_dims(space: gym.Space) -> Tuple[int, Tuple[int, ...], bool]:
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape)), tuple(space.shape), False
    if isinstance(space, gym.spaces.Discrete):
        return space.n, (1,), True
    raise ValueError("Unsupported space type for PPO agent.")


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_space: gym.Space, hidden_sizes: Tuple[int, ...], continuous: bool) -> None:
        super().__init__()
        self.continuous = continuous
        last_dim = observation_dim
        actor_layers = []
        for hidden in hidden_sizes:
            actor_layers.append(nn.Linear(last_dim, hidden))
            actor_layers.append(nn.Tanh())
            last_dim = hidden
        if continuous:
            action_dim = int(np.prod(action_space.shape))
            actor_layers.append(nn.Linear(last_dim, action_dim))
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            action_dim = action_space.n
            actor_layers.append(nn.Linear(last_dim, action_dim))
            self.log_std = None
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        last_dim = observation_dim
        for hidden in hidden_sizes:
            critic_layers.append(nn.Linear(last_dim, hidden))
            critic_layers.append(nn.Tanh())
            last_dim = hidden
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        features = observation.view(observation.size(0), -1)
        policy_output = self.actor(features)
        if self.continuous:
            std = torch.exp(self.log_std)
            distribution = Normal(policy_output, std)
        else:
            distribution = Categorical(logits=policy_output)
        value = self.critic(features).squeeze(-1)
        return distribution, value


class PPOAgent(BaseAgent):
    """On-policy PPO agent with generalized advantage estimation."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        self.device = torch.device(device)

        observation_dim = int(np.prod(observation_space.shape))
        _, action_shape, discrete_action = _infer_dims(action_space)
        self.discrete_action = discrete_action
        hidden_sizes = tuple(config.get("hidden_sizes", (64, 64)))

        self.policy = ActorCritic(observation_dim, action_space, hidden_sizes, continuous=not discrete_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))

        rollout_length = config.get("rollout_length", 2048)
        self.rollout_buffer = RolloutBuffer(
            capacity=rollout_length,
            observation_shape=tuple(observation_space.shape),
            action_shape=action_shape,
            discrete_action=discrete_action,
        )

        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.value_coef = config.get("value_coef", 0.5)
        self.update_epochs = config.get("update_epochs", 10)
        self.mini_batch_size = config.get("mini_batch_size", 64)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        self._current_step = 0
        self._last_value: torch.Tensor | None = None
        self._last_log_prob: torch.Tensor | None = None
        self._last_action: Any = None
        self._last_observation: np.ndarray | None = None

    def train_mode(self) -> None:
        self.policy.train()

    def eval_mode(self) -> None:
        self.policy.eval()

    def _prepare_observation(self, observation: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).view(1, -1)

    def select_action(self, observation, explore: bool = True) -> Any:
        obs_tensor = self._prepare_observation(observation)
        distribution, value = self.policy(obs_tensor)
        if explore:
            action_tensor = distribution.sample()
        else:
            action_tensor = distribution.mean if hasattr(distribution, "mean") else torch.argmax(distribution.logits, dim=-1)
        log_prob = distribution.log_prob(action_tensor)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(-1)
        self._last_value = value.detach()
        self._last_log_prob = log_prob.detach()
        self._last_action = action_tensor.detach()
        self._last_observation = np.asarray(observation, dtype=np.float32)

        if self.discrete_action:
            return int(action_tensor.squeeze(0).item())
        action = action_tensor.squeeze(0).cpu().numpy()
        low = self.action_space.low if isinstance(self.action_space, gym.spaces.Box) else None
        high = self.action_space.high if isinstance(self.action_space, gym.spaces.Box) else None
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def observe(self, transition: Transition) -> bool:
        if self._last_observation is None or self._last_value is None or self._last_log_prob is None:
            raise RuntimeError("`select_action` must be called before `observe`.")
        done = transition.terminated
        episode_finished = transition.done
        self.rollout_buffer.add(
            observation=self._last_observation,
            action=self._last_action.cpu().numpy() if isinstance(self._last_action, torch.Tensor) else self._last_action,
            log_prob=float(self._last_log_prob.cpu().item()),
            value=float(self._last_value.cpu().item()),
            reward=float(transition.reward),
            done=bool(done),
            next_observation=np.asarray(transition.next_observation, dtype=np.float32),
        )
        self._current_step += 1

        reached_rollout = len(self.rollout_buffer) >= self.rollout_buffer.capacity
        if episode_finished and not reached_rollout:
            # Allow bootstrapping with zero value at episode end.
            return True
        return reached_rollout

    def _evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.policy(observations)
        log_probs = distribution.log_prob(actions)
        if log_probs.ndim > 1:
            log_probs = log_probs.sum(-1)
        entropy = distribution.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(-1)
        return log_probs, entropy, values

    def _value_function(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.view(observations.size(0), -1)
        return self.policy.critic(observations)

    def update(self) -> Dict[str, Any]:
        data = self.rollout_buffer.compute_returns_and_advantages(
            value_fn=self._value_function,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        observations = data["observations"].view(data["observations"].size(0), -1)
        actions = data["actions"]
        log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        values = data["values"]

        total_loss = 0.0
        num_samples = observations.size(0)
        indices = np.arange(num_samples)
        update_steps = 0
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_log_probs = log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_log_probs, entropy, new_values = self._evaluate_actions(batch_obs, batch_actions)

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
                update_steps += 1

        self.rollout_buffer.reset()
        self._last_log_prob = None
        self._last_value = None
        self._last_observation = None
        self._last_action = None

        avg_loss = total_loss / max(1, update_steps)
        return {"loss": float(avg_loss)}
