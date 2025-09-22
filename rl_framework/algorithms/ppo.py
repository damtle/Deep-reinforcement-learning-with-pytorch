"""Proximal Policy Optimization implementation for the unified framework."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_framework.algorithms.base import BaseAgent
from rl_framework.buffers.rollout_buffer import RolloutBuffer
from rl_framework.networks.actor_critic import ActorCriticPolicy
from rl_framework.types import Transition, infer_action_spec, infer_observation_spec


class PPOAgent(BaseAgent):
    """On-policy PPO agent with generalized advantage estimation."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        self.device = torch.device(device)

        self.observation_spec = infer_observation_spec(observation_space)
        self.action_spec = infer_action_spec(action_space)
        self.discrete_action = self.action_spec.discrete
        hidden_sizes = tuple(config.get("hidden_sizes", (64, 64)))

        self.policy = ActorCriticPolicy(observation_space, action_space, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))

        rollout_length = config.get("rollout_length", 2048)
        self.rollout_buffer = RolloutBuffer(
            capacity=rollout_length,
            observation_shape=self.observation_spec.shape,
            action_shape=self.action_spec.shape,
            discrete_action=self.discrete_action,
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
        if isinstance(self._last_action, torch.Tensor):
            action_array = self._last_action.detach().cpu().numpy()
        else:
            dtype = np.int64 if self.discrete_action else np.float32
            action_array = np.asarray(self._last_action, dtype=dtype)
        if self.discrete_action:
            action_array = action_array.astype(np.int64, copy=False)
        else:
            action_array = action_array.astype(np.float32, copy=False)
        action_array = action_array.reshape(self.action_spec.shape)

        self.rollout_buffer.add(
            observation=self._last_observation,
            action=action_array,
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
        return self.policy.value_function(observations)

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
