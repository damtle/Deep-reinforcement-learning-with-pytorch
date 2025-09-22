"""Vanilla actor-critic agent."""
from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rl_framework.algorithms.base import BaseAgent
from rl_framework.networks.actor_critic import ActorCriticPolicy
from rl_framework.types import Transition


class ActorCriticAgent(BaseAgent):
    """Bootstrapped actor-critic with single-step updates."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("ActorCriticAgent requires Box observation spaces.")

        self.device = torch.device(device)
        hidden_sizes = tuple(config.get("hidden_sizes", (128, 128)))
        self.policy = ActorCriticPolicy(observation_space, action_space, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))

        self.gamma = float(config.get("gamma", 0.99))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.entropy_coef = float(config.get("entropy_coef", 0.0))
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.update_after = int(config.get("update_after", 1))

        self._trajectory: List[Dict[str, torch.Tensor]] = []
        self._last_observation: np.ndarray | None = None
        self._last_data: Dict[str, torch.Tensor] | None = None

    def train_mode(self) -> None:
        self.policy.train()

    def eval_mode(self) -> None:
        self.policy.eval()

    def _to_tensor(self, observation: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).view(1, -1)

    def _convert_action(self, action_tensor: torch.Tensor) -> Any:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(action_tensor.squeeze(0).item())
        action = action_tensor.squeeze(0).cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def select_action(self, observation, explore: bool = True) -> Any:
        obs_tensor = self._to_tensor(observation)
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
        entropy = distribution.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(-1)

        self._last_observation = np.asarray(observation, dtype=np.float32)
        self._last_data = {
            "log_prob": log_prob.squeeze(0),
            "value": value.squeeze(0),
            "entropy": entropy.squeeze(0),
        }
        return self._convert_action(action_tensor)

    def observe(self, transition: Transition) -> bool:
        if self._last_data is None or self._last_observation is None:
            raise RuntimeError("`select_action` must be called before `observe`.")
        next_observation = torch.as_tensor(np.asarray(transition.next_observation, dtype=np.float32), device=self.device)
        reward = torch.tensor(transition.reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(float(transition.done), dtype=torch.float32, device=self.device)

        self._trajectory.append(
            {
                "log_prob": self._last_data["log_prob"],
                "value": self._last_data["value"],
                "entropy": self._last_data["entropy"],
                "reward": reward,
                "done": done,
                "next_observation": next_observation,
            }
        )
        self._last_data = None
        self._last_observation = None

        return len(self._trajectory) >= self.update_after or transition.done

    def update(self) -> Dict[str, Any]:
        if not self._trajectory:
            return {}

        with torch.no_grad():
            last = self._trajectory[-1]
            if last["done"] > 0.5:
                bootstrap_value = torch.tensor(0.0, device=self.device)
            else:
                bootstrap_value = self.policy.value_function(last["next_observation"].view(1, -1)).squeeze(0)

        returns: List[torch.Tensor] = []
        next_value = bootstrap_value
        for step in reversed(self._trajectory):
            next_value = step["reward"] + self.gamma * (1.0 - step["done"]) * next_value
            returns.insert(0, next_value)

        log_probs = torch.stack([step["log_prob"] for step in self._trajectory])
        values = torch.stack([step["value"] for step in self._trajectory])
        entropies = torch.stack([step["entropy"] for step in self._trajectory])
        returns_tensor = torch.stack(returns)
        advantages = returns_tensor - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._trajectory.clear()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
