"""MDP functions for Auto_Balancing_Case environment.

This module contains observation, action, reward, and event functions specifically
designed for the Auto_Balancing_Case robot environment.
"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor


def wheel_contact_force_magnitude(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force magnitude for wheel bodies.
    
    Args:
        env: The environment instance
        sensor_cfg: SceneEntityCfg for the contact sensor
        
    Returns:
        Contact force magnitude for each wheel body, shape (num_envs, num_wheels)
    """
    # extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get the net contact forces for specified bodies
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, B, 3)
    # compute magnitude of contact forces
    force_magnitude = torch.norm(net_forces, dim=-1)  # (N, B)
    return force_magnitude.view(env.num_envs, -1)  # (N, B)


def wheel_contact_force_balance(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for balanced contact forces across all 4 wheels.
    
    This function encourages similar contact force magnitudes across all wheels,
    which helps maintain stable balancing.
    
    Args:
        env: The environment instance
        sensor_cfg: SceneEntityCfg for the contact sensor
        
    Returns:
        Reward for balanced contact forces, shape (num_envs,)
    """
    # extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get the net contact forces for all wheel bodies
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, 4, 3)
    # compute magnitude of contact forces for each wheel
    force_magnitude = torch.norm(net_forces, dim=-1)  # (N, 4)
    
    # Calculate variance of contact forces (lower variance = more balanced)
    mean_force = torch.mean(force_magnitude, dim=1, keepdim=True)  # (N, 1)
    force_variance = torch.mean((force_magnitude - mean_force) ** 2, dim=1)  # (N,)
    
    # Convert variance to reward (negative exponential to encourage low variance)
    # Add small epsilon to avoid numerical issues
    balance_reward = torch.exp(-force_variance / (mean_force.squeeze() + 1e-6))
    
    return balance_reward


def handle_contact_force_magnitude(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force magnitude for handle body.
    
    Args:
        env: The environment instance
        sensor_cfg: SceneEntityCfg for the contact sensor
        
    Returns:
        Contact force magnitude for handle body, shape (num_envs, 1)
    """
    # extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get the net contact forces (handle is a single body)
    net_forces = contact_sensor.data.net_forces_w  # (N, 1, 3)
    # compute magnitude of contact forces
    force_magnitude = torch.norm(net_forces, dim=-1)  # (N, 1)
    return force_magnitude.view(env.num_envs, -1)  # (num_envs, 1)
