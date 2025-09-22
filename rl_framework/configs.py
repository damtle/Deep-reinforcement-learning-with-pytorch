"""Configuration dataclasses and loaders for the unified RL framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EnvironmentConfig:
    """Configuration of a Gymnasium environment."""

    id: str
    max_episode_steps: Optional[int] = None
    render_mode: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration of the training loop."""

    total_episodes: int = 200
    max_steps_per_episode: Optional[int] = None
    seed: Optional[int] = None
    eval_interval: Optional[int] = None
    evaluation_episodes: int = 5
    log_interval: int = 10
    device: str = "cpu"


@dataclass
class AlgorithmConfig:
    """Algorithm name and its hyper-parameters."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig


def _build_environment_config(data: Dict[str, Any]) -> EnvironmentConfig:
    return EnvironmentConfig(
        id=data["id"],
        max_episode_steps=data.get("max_episode_steps"),
        render_mode=data.get("render_mode"),
        kwargs=data.get("kwargs", {}),
    )


def _build_algorithm_config(data: Dict[str, Any]) -> AlgorithmConfig:
    return AlgorithmConfig(name=data["name"], params=data.get("params", {}))


def _build_training_config(data: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        total_episodes=data.get("total_episodes", 200),
        max_steps_per_episode=data.get("max_steps_per_episode"),
        seed=data.get("seed"),
        eval_interval=data.get("eval_interval"),
        evaluation_episodes=data.get("evaluation_episodes", 5),
        log_interval=data.get("log_interval", 10),
        device=data.get("device", "cpu"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load configuration from a YAML file."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if "environment" not in raw_config:
        raise KeyError("Configuration must include an 'environment' section")
    if "algorithm" not in raw_config:
        raise KeyError("Configuration must include an 'algorithm' section")

    environment = _build_environment_config(raw_config["environment"])
    algorithm = _build_algorithm_config(raw_config["algorithm"])
    training = _build_training_config(raw_config.get("training", {}))

    return ExperimentConfig(environment=environment, algorithm=algorithm, training=training)
