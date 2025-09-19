"""Training loop orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

from rl_framework.algorithms.base import BaseAgent
from rl_framework.configs import EnvironmentConfig, TrainingConfig
from rl_framework.types import Transition


@dataclass
class EpisodeResult:
    episode: int
    reward: float
    length: int
    info: Dict[str, float] = field(default_factory=dict)


class Trainer:
    """Generic trainer supporting evaluation and logging."""

    def __init__(self, env_config: EnvironmentConfig, agent: BaseAgent, training_config: TrainingConfig) -> None:
        self.env_config = env_config
        self.agent = agent
        self.training_config = training_config

    def _make_env(self, seed: Optional[int] = None) -> gym.Env:
        kwargs = dict(self.env_config.kwargs)
        if self.env_config.max_episode_steps is not None:
            kwargs.setdefault("max_episode_steps", self.env_config.max_episode_steps)
        if self.env_config.render_mode is not None:
            kwargs.setdefault("render_mode", self.env_config.render_mode)
        env = gym.make(self.env_config.id, **kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return env

    def train(self) -> List[EpisodeResult]:
        results: List[EpisodeResult] = []
        env = self._make_env(seed=self.training_config.seed)
        eval_env = None
        if self.training_config.eval_interval:
            eval_env = self._make_env(seed=self.training_config.seed)

        for episode in range(1, self.training_config.total_episodes + 1):
            observation, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            self.agent.train_mode()

            max_steps = self.training_config.max_steps_per_episode
            step_limit = max_steps if max_steps is not None else np.inf
            step_count = 0
            while not done and step_count < step_limit:
                action = self.agent.select_action(observation, explore=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                manual_truncation = bool(step_limit != np.inf and (step_count + 1) >= step_limit and not (terminated or truncated))
                transition = Transition(
                    observation=np.asarray(observation, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    next_observation=np.asarray(next_observation, dtype=np.float32),
                    terminated=bool(terminated),
                    truncated=bool(truncated or manual_truncation),
                    info=info,
                )
                should_update = self.agent.observe(transition)
                if should_update:
                    self.agent.update()

                observation = next_observation
                episode_reward += reward
                episode_length += 1
                step_count += 1
                done = terminated or truncated or manual_truncation

            self.agent.on_episode_end()

            results.append(EpisodeResult(episode=episode, reward=episode_reward, length=episode_length))

            if episode % self.training_config.log_interval == 0:
                print(
                    f"Episode {episode:04d} | reward: {episode_reward:8.3f} | length: {episode_length:4d}",
                    flush=True,
                )

            if (
                eval_env is not None
                and self.training_config.eval_interval is not None
                and episode % self.training_config.eval_interval == 0
            ):
                evaluation = self.evaluate(eval_env)
                avg_reward = np.mean([r.reward for r in evaluation]) if evaluation else float("nan")
                print(f"Evaluation after episode {episode}: avg reward {avg_reward:.3f}")

        env.close()
        if eval_env is not None:
            eval_env.close()
        return results

    def evaluate(self, env: Optional[gym.Env] = None) -> List[EpisodeResult]:
        created_env = False
        if env is None:
            env = self._make_env(seed=self.training_config.seed)
            created_env = True
        self.agent.eval_mode()
        results: List[EpisodeResult] = []
        episodes = self.training_config.evaluation_episodes
        for episode in range(episodes):
            observation, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            while not done:
                action = self.agent.select_action(observation, explore=False)
                next_observation, reward, terminated, truncated, info = env.step(action)
                observation = next_observation
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated
            results.append(EpisodeResult(episode=episode, reward=ep_reward, length=ep_length))
        if created_env:
            env.close()
        return results
