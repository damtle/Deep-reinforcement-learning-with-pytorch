"""Shared actor-critic network components."""
from __future__ import annotations

from typing import Tuple, Type

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from rl_framework.networks.mlp import build_mlp


class ActorCriticPolicy(nn.Module):
    """A flexible actor-critic model supporting discrete and continuous actions."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Tuple[int, ...],
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("ActorCriticPolicy currently only supports Box observation spaces.")

        self.observation_dim = int(np.prod(observation_space.shape))
        self.continuous = isinstance(action_space, gym.spaces.Box)

        if self.continuous:
            self.action_dim = int(np.prod(action_space.shape))
            self.actor = build_mlp(self.observation_dim, self.action_dim, hidden_sizes, activation=activation)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.actor = build_mlp(self.observation_dim, self.action_dim, hidden_sizes, activation=activation)
            self.log_std = None
        else:
            raise TypeError("Unsupported action space for ActorCriticPolicy.")

        self.critic = build_mlp(self.observation_dim, 1, hidden_sizes, activation=activation)

    def forward(self, observation: torch.Tensor):
        observation = observation.view(observation.size(0), -1)
        policy_output = self.actor(observation)
        if self.continuous:
            std = torch.exp(self.log_std).expand_as(policy_output)
            distribution = Normal(policy_output, std)
        else:
            distribution = Categorical(logits=policy_output)
        value = self.critic(observation).squeeze(-1)
        return distribution, value

    def value_function(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.view(observation.size(0), -1)
        return self.critic(observation).squeeze(-1)

