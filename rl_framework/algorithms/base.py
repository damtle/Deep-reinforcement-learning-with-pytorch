"""Base agent abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import gymnasium as gym

from rl_framework.types import Transition


class BaseAgent(ABC):
    """Abstract base class that all agents must inherit from."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = device

    def train_mode(self) -> None:
        """Switch the internal networks to training mode."""

    def eval_mode(self) -> None:
        """Switch the internal networks to evaluation mode."""

    @abstractmethod
    def select_action(self, observation, explore: bool = True) -> Any:
        """Select an action for the given observation."""

    @abstractmethod
    def observe(self, transition: Transition) -> bool:
        """Consume a transition and return whether an update should be triggered."""

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        """Perform a learning update and return diagnostic metrics."""

    def on_episode_end(self) -> None:
        """Hook called at the end of each episode."""

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - optional persistence
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # pragma: no cover - optional persistence
        raise NotImplementedError
