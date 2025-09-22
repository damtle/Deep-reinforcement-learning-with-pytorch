"""Advantage Actor-Critic (synchronous) agent."""
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.rollout_buffer import RolloutBuffer
from rl_framework.networks.actor_critic import ActorCriticPolicy
from rl_framework.types import Transition


class A2CAgent(BaseAgent):
    """Synchronous n-step Advantage Actor-Critic agent."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("A2C requires Box observation spaces.")

        self.device = torch.device(device)
        hidden_sizes = tuple(config.get("hidden_sizes", (128, 128)))
        self.policy = ActorCriticPolicy(observation_space, action_space, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 7e-4))

        self.n_steps = int(config.get("n_steps", 5))
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 1.0))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        action_shape = action_space.shape if isinstance(action_space, gym.spaces.Box) else (1,)
        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)
        self.rollout_buffer = RolloutBuffer(
            capacity=self.n_steps,
            observation_shape=tuple(observation_space.shape),
            action_shape=action_shape,
            discrete_action=self.discrete_action,
        )

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

    def _convert_action(self, action_tensor: torch.Tensor) -> Any:
        if self.discrete_action:
            return int(action_tensor.squeeze(0).item())
        action = action_tensor.squeeze(0).cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def select_action(self, observation, explore: bool = True) -> Any:
        obs_tensor = self._prepare_observation(observation)
        distribution, value = self.policy(obs_tensor)
        if explore:
            action_tensor = distribution.sample()
        else:
            if hasattr(distribution, "mean"):
                action_tensor = distribution.mean
            else:
                action_tensor = torch.argmax(distribution.logits, dim=-1)
        log_prob = distribution.log_prob(action_tensor)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(-1)

        self._last_value = value.detach()
        self._last_log_prob = log_prob.detach()
        self._last_action = action_tensor.detach()
        self._last_observation = np.asarray(observation, dtype=np.float32)
        return self._convert_action(action_tensor)

    def observe(self, transition: Transition) -> bool:
        if self._last_observation is None or self._last_log_prob is None or self._last_value is None:
            raise RuntimeError("`select_action` must be called before `observe`.")

        done = bool(transition.done)
        self.rollout_buffer.add(
            observation=self._last_observation,
            action=self._last_action.cpu().numpy() if isinstance(self._last_action, torch.Tensor) else self._last_action,
            log_prob=float(self._last_log_prob.cpu().item()),
            value=float(self._last_value.cpu().item()),
            reward=float(transition.reward),
            done=float(transition.terminated),
            next_observation=np.asarray(transition.next_observation, dtype=np.float32),
        )

        self._last_value = None
        self._last_log_prob = None
        self._last_action = None
        self._last_observation = None

        return len(self.rollout_buffer) >= self.n_steps or done

    def _evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        distribution, values = self.policy(observations)
        log_probs = distribution.log_prob(actions)
        if log_probs.ndim > 1:
            log_probs = log_probs.sum(-1)
        entropy = distribution.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(-1)
        return log_probs, entropy, values

    def _value_function(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy.value_function(observations)

    def update(self) -> Dict[str, Any]:
        if len(self.rollout_buffer) == 0:
            return {}

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

        new_log_probs, entropy, values = self._evaluate_actions(observations, actions)

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = entropy.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.rollout_buffer.reset()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
