"""Actor-Critic with Experience Replay (ACER) agent."""
from __future__ import annotations

import random
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rl_framework.algorithms.base import BaseAgent
from rl_framework.networks.mlp import build_mlp
from rl_framework.types import Transition


class ACERNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = observation_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.q_head = nn.Linear(last_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observation = observation.view(observation.size(0), -1)
        features = self.shared(observation)
        logits = self.policy_head(features)
        q_values = self.q_head(features)
        return logits, q_values


class ACERAgent(BaseAgent):
    """Discrete ACER agent with trajectory replay."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: Dict[str, Any], device: str = "cpu") -> None:
        super().__init__(observation_space, action_space, config, device)
        if not isinstance(observation_space, gym.spaces.Box) or not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("ACERAgent requires Box observations and discrete actions.")

        self.device = torch.device(device)
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = action_space.n
        hidden_sizes = tuple(config.get("hidden_sizes", (256, 256)))

        self.network = ACERNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.get("lr", 7e-4))

        self.gamma = float(config.get("gamma", 0.99))
        self.truncation_clip = float(config.get("truncation_clip", 10.0))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.replay_ratio = int(config.get("replay_ratio", 4))
        self.max_replay_size = int(config.get("replay_buffer_size", 5000))
        self.rollout_length = int(config.get("rollout_length", 32))

        self.current_trajectory: List[Dict[str, Any]] = []
        self.pending: List[Dict[str, np.ndarray]] = []
        self.replay_buffer: List[Dict[str, np.ndarray]] = []

        self._last_logits: torch.Tensor | None = None
        self._last_action: int | None = None
        self._last_observation: np.ndarray | None = None

    def train_mode(self) -> None:
        self.network.train()

    def eval_mode(self) -> None:
        self.network.eval()

    def _tensor_observation(self, observation: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).view(1, -1)

    def select_action(self, observation, explore: bool = True) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        obs_tensor = self._tensor_observation(obs)
        with torch.no_grad():
            logits, q_values = self.network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
        if explore:
            action_tensor = torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            action_tensor = torch.argmax(probs, dim=-1)
        action = int(action_tensor.item())

        self._last_logits = logits.detach()
        self._last_action = action
        self._last_observation = obs
        return action

    def _finalize_trajectory(self) -> Dict[str, np.ndarray]:
        observations = np.stack([step["observation"] for step in self.current_trajectory])
        actions = np.asarray([step["action"] for step in self.current_trajectory], dtype=np.int64)
        rewards = np.asarray([step["reward"] for step in self.current_trajectory], dtype=np.float32)
        dones = np.asarray([step["done"] for step in self.current_trajectory], dtype=np.float32)
        next_observations = np.stack([step["next_observation"] for step in self.current_trajectory])
        behavior_logits = np.stack([step["behavior_logits"] for step in self.current_trajectory])
        self.current_trajectory.clear()
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_observations": next_observations,
            "behavior_logits": behavior_logits,
        }

    def observe(self, transition: Transition) -> bool:
        if self._last_logits is None or self._last_action is None or self._last_observation is None:
            raise RuntimeError("`select_action` must be called before `observe`.")

        entry = {
            "observation": self._last_observation,
            "action": int(self._last_action),
            "reward": float(transition.reward),
            "done": float(transition.terminated),
            "next_observation": np.asarray(transition.next_observation, dtype=np.float32),
            "behavior_logits": self._last_logits.squeeze(0).cpu().numpy(),
        }
        self.current_trajectory.append(entry)
        self._last_logits = None
        self._last_action = None
        self._last_observation = None

        if transition.done or len(self.current_trajectory) >= self.rollout_length:
            trajectory = self._finalize_trajectory()
            self.pending.append(trajectory)
            self.replay_buffer.append(trajectory)
            if len(self.replay_buffer) > self.max_replay_size:
                self.replay_buffer.pop(0)
        return len(self.pending) > 0

    def _policy_value(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, q_values = self.network(observation)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        values = (probs * q_values).sum(dim=-1)
        return probs, log_probs, values

    def _apply_update(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, float]:
        observations = torch.as_tensor(trajectory["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(trajectory["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(trajectory["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(trajectory["dones"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(trajectory["next_observations"], dtype=torch.float32, device=self.device)
        behavior_logits = torch.as_tensor(trajectory["behavior_logits"], dtype=torch.float32, device=self.device)

        probs, log_probs, values = self._policy_value(observations)
        q_logits, q_values = self.network(observations)
        policy_probs = torch.softmax(q_logits, dim=-1)
        q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        pi_a = policy_probs.gather(1, actions.view(-1, 1)).squeeze(1)

        mu = torch.softmax(behavior_logits, dim=-1)
        mu_a = mu.gather(1, actions.view(-1, 1)).squeeze(1)
        rho = pi_a / (mu_a + 1e-8)
        rho_clipped = torch.clamp(rho, max=self.truncation_clip)

        entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=1)
        log_pi_a = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_probs, _, next_values = self._policy_value(next_observations)
            q_ret = next_values[-1]
            if dones[-1] > 0.5:
                q_ret = torch.zeros_like(q_ret)

        policy_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []

        for idx in reversed(range(len(rewards))):
            q_ret = rewards[idx] + self.gamma * (1 - dones[idx]) * q_ret
            advantage = q_ret - values[idx]
            policy_losses.append(-rho_clipped[idx].detach() * log_pi_a[idx] * advantage.detach())
            value_losses.append((q_selected[idx] - q_ret.detach()).pow(2))
            entropy_terms.append(entropy[idx])
            q_ret = (rho_clipped[idx].detach() * (q_ret - q_selected[idx].detach()) + values[idx]).detach()

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_terms).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=40.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def update(self) -> Dict[str, Any]:
        if not self.pending:
            return {}

        metrics: Dict[str, float] = {}
        while self.pending:
            trajectory = self.pending.pop(0)
            metrics = self._apply_update(trajectory)
            for _ in range(self.replay_ratio):
                if not self.replay_buffer:
                    break
                replay_traj = random.choice(self.replay_buffer)
                metrics = self._apply_update(replay_traj)
        return metrics
