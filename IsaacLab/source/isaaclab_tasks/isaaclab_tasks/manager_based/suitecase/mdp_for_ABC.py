"""MDP functions for Auto_Balancing_Case environment.

This module contains observation, action, reward, and event functions specifically
designed for the Auto_Balancing_Case robot environment.
"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor

# observation functions

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


# reward functions

def wheel_contact_force_variance(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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


def wheel_contact_force_min_max(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the difference between maximum and minimum wheel contact forces.
    
    This function penalizes large differences between the highest and lowest
    contact forces across all wheels, encouraging more uniform force distribution.
    
    Args:
        env: The environment instance
        sensor_cfg: SceneEntityCfg for the contact sensor
        
    Returns:
        Penalty for unbalanced contact forces, shape (num_envs,)
    """
    # extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get the net contact forces for all wheel bodies
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, 4, 3)
    # compute magnitude of contact forces for each wheel
    force_magnitude = torch.norm(net_forces, dim=-1)  # (N, 4)
    
    # Calculate max and min contact forces for each environment
    max_force = torch.max(force_magnitude, dim=1)[0]  # (N,)
    min_force = torch.min(force_magnitude, dim=1)[0]  # (N,)
    
    # Calculate the difference between max and min forces
    force_diff = max_force - min_force  # (N,)

    return force_diff

def desired_contacts_any(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).any(dim=1)
    return 1.0 * zero_contact


#event functions

def apply_external_force_torque_offset(
    env,
    env_ids,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply external forces and torques with position offset to the bodies.
    
    This function creates a set of random forces and torques sampled from the given ranges.
    The forces and torques are applied to the bodies with a specified position offset from
    the body's center of mass. This allows applying wrench at specific points on the body.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to apply the wrench to
        force_range: Range for force magnitude (min, max)
        torque_range: Range for torque magnitude (min, max)
        position_offset: Position offset from body center of mass (x, y, z)
        asset_cfg: SceneEntityCfg for the asset to apply wrench to
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = torch.rand(size, device=asset.device) * (force_range[1] - force_range[0]) + force_range[0]
    torques = torch.rand(size, device=asset.device) * (torque_range[1] - torque_range[0]) + torque_range[0]
    
    # Convert position offset to tensor
    position_offset_tensor = torch.tensor(position_offset, device=asset.device, dtype=torch.float32)
    position_offset_tensor = position_offset_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    position_offset_tensor = position_offset_tensor.expand(len(env_ids), num_bodies, 3)  # (N, B, 3)
    
    # Use the built-in set_external_force_and_torque function with positions parameter
    asset.set_external_force_and_torque(
        forces=forces, 
        torques=torques, 
        positions=position_offset_tensor,
        body_ids=asset_cfg.body_ids, 
        env_ids=env_ids,
        is_global=False  # Apply in world frame
    )

def apply_specific_external_force_torque(
    env,
    env_ids,
    force_x_range: tuple[float, float] = (0.0, 0.0),  # x-direction force range
    force_y: float = 0.0,  # y-direction force (fixed)
    force_z: float = 0.0,  # z-direction force (fixed)
    torque_x_range: tuple[float, float] = (0.0, 0.0),  # x-direction torque range
    torque_y: float = 0.0,  # y-direction torque (fixed)
    torque_z: float = 0.0,  # z-direction torque (fixed)
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply external forces and torques with x-direction ranges to the bodies.
    
    This function applies forces and torques where x-direction values are sampled from ranges,
    while y and z directions use fixed values. All values are applied in world frame.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to apply the wrench to
        force_x_range: Range for x-direction force (min, max)
        force_y: Fixed y-direction force
        force_z: Fixed z-direction force
        torque_x_range: Range for x-direction torque (min, max)
        torque_y: Fixed y-direction torque
        torque_z: Fixed z-direction torque
        position_offset: Position offset from body center of mass (x, y, z)
        asset_cfg: SceneEntityCfg for the asset to apply wrench to
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Sample random x-direction forces and torques from ranges
    size = (len(env_ids), num_bodies)
    force_x = torch.rand(size, device=asset.device) * (force_x_range[1] - force_x_range[0]) + force_x_range[0]
    torque_x = torch.rand(size, device=asset.device) * (torque_x_range[1] - torque_x_range[0]) + torque_x_range[0]
    
    # Create force and torque tensors
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    torques = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    
    # Set x-direction values (sampled from ranges)
    forces[:, :, 0] = force_x
    torques[:, :, 0] = torque_x
    
    # Set y and z direction values (fixed)
    forces[:, :, 1] = force_y
    forces[:, :, 2] = force_z
    torques[:, :, 1] = torque_y
    torques[:, :, 2] = torque_z
    
    # Convert position offset to tensor
    position_offset_tensor = torch.tensor(position_offset, device=asset.device, dtype=torch.float32)
    position_offset_tensor = position_offset_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    position_offset_tensor = position_offset_tensor.expand(len(env_ids), num_bodies, 3)  # (N, B, 3)
    
    # Use the built-in set_external_force_and_torque function with positions parameter
    asset.set_external_force_and_torque(
        forces=forces, 
        torques=torques, 
        positions=position_offset_tensor,
        body_ids=asset_cfg.body_ids, 
        env_ids=env_ids,
        is_global=False  # Apply in world frame
    )

def push_by_setting_specific_velocity(
    env,
    env_ids,
    vel_x_range: tuple[float, float] = (0.0, 0.0),  # x-direction velocity range
    vel_y: float = 0.0,  # y-direction velocity (fixed)
    vel_z: float = 0.0,  # z-direction velocity (fixed)
    ang_vel_x: float = 0.0,  # x-axis angular velocity (roll, fixed)
    ang_vel_y: float = 0.0,  # y-axis angular velocity (pitch, fixed)
    ang_vel_z: float = 0.0,  # z-axis angular velocity (yaw, fixed)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting root velocity values with x-direction range.
    
    This function sets velocity values where x-direction linear velocity is sampled from a range,
    while other directions use fixed values. All values are applied in world frame.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to apply the velocity to
        vel_x_range: Range for x-direction linear velocity (min, max)
        vel_y: Fixed y-direction linear velocity
        vel_z: Fixed z-direction linear velocity
        ang_vel_x: Fixed x-axis angular velocity (roll)
        ang_vel_y: Fixed y-axis angular velocity (pitch)
        ang_vel_z: Fixed z-axis angular velocity (yaw)
        asset_cfg: SceneEntityCfg for the asset to apply velocity to
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    
    # Sample random x-direction velocity from range
    vel_x = torch.rand(len(env_ids), device=asset.device) * (vel_x_range[1] - vel_x_range[0]) + vel_x_range[0]
    
    # Create velocity tensor [x, y, z, roll, pitch, yaw]
    velocities = torch.zeros(len(env_ids), 6, device=asset.device)
    
    # Set x-direction velocity (sampled from range)
    velocities[:, 0] = vel_x
    
    # Set other direction velocities (fixed)
    velocities[:, 1] = vel_y  # y-direction
    velocities[:, 2] = vel_z  # z-direction
    velocities[:, 3] = ang_vel_x  # roll
    velocities[:, 4] = ang_vel_y  # pitch
    velocities[:, 5] = ang_vel_z  # yaw
    
    # Get current velocities and add the specified velocities
    vel_w = asset.data.root_vel_w[env_ids].clone()
    vel_w += velocities
    
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def clear_external_force_torque(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Clear external forces and torques from the bodies.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to clear the wrench from
        asset_cfg: SceneEntityCfg for the asset to clear wrench from
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Create zero tensors to clear the wrench
    zero_forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    zero_torques = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    
    # Clear the external wrench
    asset.set_external_force_and_torque(
        forces=zero_forces, 
        torques=zero_torques, 
        body_ids=asset_cfg.body_ids, 
        env_ids=env_ids
    )
