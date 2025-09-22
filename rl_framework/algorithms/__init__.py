"""Algorithm registry for the unified RL framework."""
from __future__ import annotations

from typing import Dict, Type

import gymnasium as gym

from rl_framework.algorithms.a2c import A2CAgent
from rl_framework.algorithms.acer import ACERAgent
from rl_framework.algorithms.actor_critic import ActorCriticAgent
from rl_framework.algorithms.base import BaseAgent
from rl_framework.algorithms.ddpg import DDPGAgent
from rl_framework.algorithms.dqn import DQNAgent
from rl_framework.algorithms.policy_gradient import PolicyGradientAgent
from rl_framework.algorithms.ppo import PPOAgent
from rl_framework.algorithms.sac import SACAgent
from rl_framework.algorithms.td3 import TD3Agent


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "dqn": DQNAgent,
    "policy_gradient": PolicyGradientAgent,
    "pg": PolicyGradientAgent,
    "actor_critic": ActorCriticAgent,
    "a2c": A2CAgent,
    "ddpg": DDPGAgent,
    "ppo": PPOAgent,
    "acer": ACERAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
}


def make_agent(name: str, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, object], device: str) -> BaseAgent:
    try:
        agent_cls = AGENT_REGISTRY[name.lower()]
    except KeyError as err:  # pragma: no cover - sanity guard
        raise KeyError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY)}") from err
    return agent_cls(observation_space=observation_space, action_space=action_space, config=config, device=device)