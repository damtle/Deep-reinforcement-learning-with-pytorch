"""Shared type definitions for the unified RL framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class Transition:
    """Container for one environment interaction."""

    observation: np.ndarray
    action: Any
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class SpaceSpec:
    """Simple description of a Gymnasium space."""

    shape: Tuple[int, ...]
    dim: int
    discrete: bool


def infer_observation_spec(space: gym.Space) -> SpaceSpec:
    """Infer the shape information for an observation space."""

    if isinstance(space, gym.spaces.Box):
        shape = tuple(space.shape)
        return SpaceSpec(shape=shape, dim=int(np.prod(shape)), discrete=False)
    raise TypeError("Only Box observation spaces are supported by the current agents.")


def infer_action_spec(space: gym.Space) -> SpaceSpec:
    """Infer the actionable shape/size information for an action space."""

    if isinstance(space, gym.spaces.Box):
        shape = tuple(space.shape)
        return SpaceSpec(shape=shape, dim=int(np.prod(shape)), discrete=False)
    if isinstance(space, gym.spaces.Discrete):
        return SpaceSpec(shape=(1,), dim=space.n, discrete=True)
    raise TypeError("Unsupported action space type for the current agents.")
