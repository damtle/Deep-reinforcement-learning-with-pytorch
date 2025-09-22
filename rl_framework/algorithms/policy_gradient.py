"""REINFORCE style policy gradient agent."""
from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rl_framework.algorithms.base import BaseAgent
from rl_framework.networks.mlp import build_mlp
from rl_framework.types import Transition


class PolicyGradientAgent(BaseAgent):
    """Monte-Carlo policy gradient (REINFORCE) agent."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("PolicyGradientAgent currently only supports Box observation spaces.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("PolicyGradientAgent requires a discrete action space.")

        self.device = torch.device(device)
        obs_dim = int(np.prod(observation_space.shape))
        hidden_sizes = tuple(config.get("hidden_sizes", (128, 128)))
        self.policy = build_mlp(obs_dim, action_space.n, hidden_sizes, activation=nn.ReLU).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))

        self.gamma = float(config.get("gamma", 0.99))
        self.normalize_returns = bool(config.get("normalize_returns", True))
        self.entropy_coef = float(config.get("entropy_coef", 0.0))

        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[torch.Tensor] = []
        self._entropies: List[torch.Tensor] = []

    def train_mode(self) -> None:
        self.policy.train()

    def eval_mode(self) -> None:
        self.policy.eval()

    def _distribution(self, observation: np.ndarray) -> Categorical:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        logits = self.policy(obs_tensor)
        return Categorical(logits=logits)

    def select_action(self, observation, explore: bool = True) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        distribution = self._distribution(observation)
        if explore:
            action_tensor = distribution.sample()
        else:
            action_tensor = torch.argmax(distribution.logits, dim=-1)
        log_prob = distribution.log_prob(action_tensor)
        entropy = distribution.entropy()

        self._log_probs.append(log_prob.squeeze(0))
        self._entropies.append(entropy.squeeze(0))
        return int(action_tensor.item())

    def observe(self, transition: Transition) -> bool:
        self._rewards.append(torch.tensor(transition.reward, dtype=torch.float32, device=self.device))
        return transition.done

    def update(self) -> Dict[str, Any]:
        if not self._rewards:
            return {}

        returns: List[torch.Tensor] = []
        cumulative = torch.tensor(0.0, device=self.device)
        for reward in reversed(self._rewards):
            cumulative = reward + self.gamma * cumulative
            returns.insert(0, cumulative)
        returns_tensor = torch.stack(returns)
        if self.normalize_returns:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)

        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)

        loss = -(log_probs * returns_tensor.detach()).mean() - self.entropy_coef * entropies.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        episode_return = float(sum(r.item() for r in self._rewards))

        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()

        return {"loss": float(loss.item()), "episode_return": episode_return}
