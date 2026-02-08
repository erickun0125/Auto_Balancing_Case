#!/usr/bin/env python3
"""
Policy Interface for RSL-RL ActorCritic Models

Loads and runs inference on PPO policies trained in Isaac Lab using the RSL-RL
framework. Handles observation history construction (4-step x 8-dim = 32-dim)
and provides deterministic action prediction for real-time deployment.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the RSL-RL policy network.

    Network architecture must match the training configuration exactly.
    Default values correspond to the trained Auto Balancing Case model.
    """
    model_path: str
    device: str = "cpu"
    obs_history_length: int = 4

    # RSL-RL ActorCritic network architecture
    actor_hidden_dims: Optional[List[int]] = None
    critic_hidden_dims: Optional[List[int]] = None
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"

    # Observation dimensions (matches Isaac Lab environment)
    obs_dim_per_timestep: int = 8   # joint_pos(1) + joint_vel(1) + prev_action(1) + wheel_forces(4) + handle_force(1)
    num_actions: int = 1            # Single balance joint

    def __post_init__(self) -> None:
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 128, 64]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 128, 64]


class RSLRLActorCritic(nn.Module):
    """RSL-RL ActorCritic network replica for inference.

    Mirrors the exact architecture used during training so that
    checkpoint state_dicts can be loaded directly.

    Architecture: Input -> [hidden1, hidden2, hidden3] -> Output
    with ELU activation between layers.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

        num_actor_obs = config.obs_dim_per_timestep * config.obs_history_length
        num_critic_obs = num_actor_obs
        num_actions = config.num_actions

        actor_hidden_dims = config.actor_hidden_dims or [256, 128, 64]
        critic_hidden_dims = config.critic_hidden_dims or [256, 128, 64]
        activation = self._get_activation(config.activation)

        # Actor network
        actor_layers: List[nn.Module] = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers: List[nn.Module] = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise (matches RSL-RL training distribution)
        self.noise_std_type = config.noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(config.init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(config.init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation name ('elu', 'relu', 'tanh').

        Returns:
            PyTorch activation module.

        Raises:
            ValueError: If activation name is unknown.
        """
        activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
        }
        cls = activations.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown activation: '{name}'. Supported: {list(activations.keys())}")
        return cls()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action prediction (no exploration noise).

        Args:
            observations: Batch of observations [batch_size, obs_dim].

        Returns:
            Action tensor [batch_size, num_actions].
        """
        return self.actor(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass (inference mode)."""
        return self.act_inference(observations)


class PolicyInterface:
    """High-level interface for loading and running RSL-RL policies.

    Handles checkpoint loading, observation history formatting, and
    deterministic inference for real-time deployment.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self.policy: Optional[RSLRLActorCritic] = None
        self._load_policy()

    def _load_policy(self) -> None:
        """Load the trained policy from an RSL-RL checkpoint file.

        Supports both direct state_dict and wrapped checkpoint formats
        (with 'model_state_dict' key).

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If checkpoint loading fails.
        """
        logger.info("Loading policy from %s", self.config.model_path)

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Policy file not found: {self.config.model_path}")

        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
            logger.info("Checkpoint keys: %s", list(checkpoint.keys()))

            # RSL-RL stores the full ActorCritic state under 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            self.policy = RSLRLActorCritic(self.config)
            self.policy.load_state_dict(model_state_dict, strict=True)
            self.policy.eval()

            logger.info("RSL-RL ActorCritic model loaded successfully")
            logger.info("  Actor: %s", self.policy.actor)
            logger.info("  Critic: %s", self.policy.critic)
            logger.info("  Total parameters: %s", f"{sum(p.numel() for p in self.policy.parameters()):,}")

        except Exception as e:
            logger.error("Policy loading error: %s", e)
            raise

    def prepare_policy_input(self, obs_history: List[Dict[str, np.ndarray]]) -> torch.Tensor:
        """Convert observation history to a flat policy input tensor.

        Each timestep observation is concatenated into an 8-dim vector:
        [joint_pos, joint_vel, prev_action, wheel_forces(4), handle_force].
        The history is concatenated oldest-first: 8 x 4 = 32 dimensions.

        If history is shorter than obs_history_length, the oldest slots
        are zero-padded. If longer, only the most recent entries are used.

        Args:
            obs_history: List of observation dictionaries.

        Returns:
            Policy input tensor of shape [1, obs_dim_per_timestep * obs_history_length].

        Raises:
            ValueError: If observation history is empty.
        """
        if len(obs_history) == 0:
            raise ValueError("Observation history is empty")

        obs_vectors: List[np.ndarray] = []
        for obs in obs_history:
            obs_vector = np.concatenate([
                obs['joint_pos'],               # 1 dim
                obs['joint_vel'],               # 1 dim
                obs['prev_action'],             # 1 dim
                obs['wheel_contact_forces'],    # 4 dims
                obs['handle_external_force']    # 1 dim
            ])  # 8 dims per timestep
            obs_vectors.append(obs_vector)

        # Zero-pad oldest slots if history is too short
        while len(obs_vectors) < self.config.obs_history_length:
            obs_vectors.insert(0, np.zeros(self.config.obs_dim_per_timestep))

        # Keep only the most recent entries
        if len(obs_vectors) > self.config.obs_history_length:
            obs_vectors = obs_vectors[-self.config.obs_history_length:]

        # Concatenate: 8 dims x 4 timesteps = 32 dims
        policy_input = np.concatenate(obs_vectors)
        policy_input = torch.from_numpy(policy_input).float().unsqueeze(0)

        return policy_input.to(self.config.device)

    def predict(self, obs_history: List[Dict[str, np.ndarray]]) -> float:
        """Run deterministic policy inference.

        Args:
            obs_history: List of observation dictionaries (at least 1 entry).

        Returns:
            Scalar action value (radians).

        Raises:
            RuntimeError: If policy is not loaded.
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded")

        policy_input = self.prepare_policy_input(obs_history)

        with torch.no_grad():
            action_tensor = self.policy.act_inference(policy_input)
            action = action_tensor.squeeze().cpu().numpy()
            return float(action) if action.ndim == 0 else float(action[0])

    def get_value(self, obs_history: List[Dict[str, np.ndarray]]) -> float:
        """Run critic value prediction (for debugging/monitoring).

        Args:
            obs_history: List of observation dictionaries.

        Returns:
            Scalar value estimate.
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded")

        policy_input = self.prepare_policy_input(obs_history)

        with torch.no_grad():
            value_tensor = self.policy.critic(policy_input)
            value = value_tensor.squeeze().cpu().numpy()
            return float(value) if value.ndim == 0 else float(value[0])

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for logging/debugging.

        Returns:
            Dictionary with model architecture and status information.
        """
        if self.policy is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "actor_hidden_dims": self.config.actor_hidden_dims,
            "critic_hidden_dims": self.config.critic_hidden_dims,
            "activation": self.config.activation,
            "obs_dim": self.config.obs_dim_per_timestep * self.config.obs_history_length,
            "num_actions": self.config.num_actions,
            "device": self.config.device,
            "total_parameters": sum(p.numel() for p in self.policy.parameters())
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    config = PolicyConfig(model_path="trained_model/model_49999.pt", device="cpu")
    policy_interface = PolicyInterface(config)

    logger.info("Model info: %s", policy_interface.get_model_info())

    dummy_obs = {
        'joint_pos': np.array([0.1]),
        'joint_vel': np.array([0.05]),
        'prev_action': np.array([0.0]),
        'wheel_contact_forces': np.array([10.0, 12.0, 11.0, 13.0]),
        'handle_external_force': np.array([2.0])
    }
    obs_history = [dummy_obs] * 4

    try:
        action = policy_interface.predict(obs_history)
        value = policy_interface.get_value(obs_history)
        logger.info("Test prediction — Action: %.4f, Value: %.4f", action, value)
    except Exception as e:
        logger.error("Test failed: %s", e)
