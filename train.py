"""Command line entry point for the unified RL framework."""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_framework.algorithms import make_agent
from rl_framework.configs import load_config
from rl_framework.trainers.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified reinforcement learning training script.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--total-episodes", type=int, help="Override the number of training episodes.")
    parser.add_argument("--device", type=str, help="Override the device to run on (cpu, cuda, ...).")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    return parser.parse_args()


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _parse_args()
    config = load_config(args.config)

    if args.total_episodes is not None:
        config.training.total_episodes = args.total_episodes
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    if config.training.seed is not None:
        _set_global_seed(config.training.seed)

    env_preview = gym.make(
        config.environment.id,
        **{
            **config.environment.kwargs,
            **({"max_episode_steps": config.environment.max_episode_steps} if config.environment.max_episode_steps else {}),
            **({"render_mode": config.environment.render_mode} if config.environment.render_mode else {}),
        },
    )
    observation_space = env_preview.observation_space
    action_space = env_preview.action_space
    env_preview.close()

    agent = make_agent(
        name=config.algorithm.name,
        observation_space=observation_space,
        action_space=action_space,
        config=config.algorithm.params,
        device=config.training.device,
    )

    trainer = Trainer(env_config=config.environment, agent=agent, training_config=config.training)
    trainer.train()


if __name__ == "__main__":
    main()
