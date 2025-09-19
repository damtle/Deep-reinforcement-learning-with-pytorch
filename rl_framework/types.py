"""Shared type definitions for the unified RL framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
