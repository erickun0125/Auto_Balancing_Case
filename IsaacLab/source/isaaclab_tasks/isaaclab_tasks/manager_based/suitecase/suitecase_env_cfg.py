"""Suitcase wheeled robot: manager-based RL environment for auto-balancing with USD asset.

This environment loads a user-provided USD and trains a policy to keep the body balanced against
external disturbances while maintaining ground contact on four wheels. The single action is the
desired position of the hinge joint between the bottom plane and the base body.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.modifiers import ModifierCfg

# Common MDP utilities
from isaaclab.envs import mdp
from isaaclab.actuators import ImplicitActuatorCfg

# Custom MDP functions for Auto_Balancing_Case
from .mdp_for_ABC import (
    wheel_contact_force_magnitude, 
    handle_contact_force_magnitude, 
    wheel_contact_force_variance,
    wheel_contact_force_min_max,
    apply_external_force_torque_offset,
    apply_specific_external_force_torque,
    push_by_setting_specific_velocity,
    nan_detection_termination,
    extreme_values_termination
)

import torch


def safe_nan_replacement(data: torch.Tensor, replacement_value: float = 0.0, method: str = "zero", 
                        obs_name: str = "unknown", enable_logging: bool = True, max_log_freq: int = 100) -> torch.Tensor:
    """NaN 값을 안전한 값으로 대체하는 modifier 함수.
    
    Args:
        data: 입력 텐서
        replacement_value: NaN을 대체할 값 (기본값: 0.0)
        method: NaN 대체 방법 ("zero", "mean", "previous")
        obs_name: observation 이름 (디버깅용)
        enable_logging: NaN 발생 시 로깅 활성화 (기본값: False)
        max_log_freq: 최대 로깅 빈도 (기본값: 100)
        
    Returns:
        NaN이 안전한 값으로 대체된 텐서
    """
    if not torch.isnan(data).any():
        return data
    
    # NaN 개수 및 위치 로깅 (빈도 제한)
    nan_count = torch.isnan(data).sum().item()
    if enable_logging and nan_count > 0:
        # 로깅 빈도 제한을 위한 간단한 카운터
        if not hasattr(safe_nan_replacement, '_log_counter'):
            safe_nan_replacement._log_counter = {}
        if obs_name not in safe_nan_replacement._log_counter:
            safe_nan_replacement._log_counter[obs_name] = 0
        
        if safe_nan_replacement._log_counter[obs_name] < max_log_freq:
            print(f"[NaN Safety] {obs_name}: Detected {nan_count} NaN values in tensor of shape {data.shape}")
            print(f"[NaN Safety] {obs_name}: Using method: {method}, replacement value: {replacement_value}")
            safe_nan_replacement._log_counter[obs_name] += 1
    
    # NaN이 있는 경우에만 처리 (성능 최적화)
    if method == "zero":
        # 0.0으로 대체
        data = torch.where(torch.isnan(data), torch.tensor(replacement_value, device=data.device, dtype=data.dtype), data)
    elif method == "mean":
        # 각 차원별로 평균값 계산 (NaN 제외)
        # batch dimension을 제외하고 계산
        if data.dim() > 1:
            # 각 환경별로 평균 계산 (벡터화된 연산 사용)
            for env_idx in range(data.shape[0]):
                env_data = data[env_idx]
                if torch.isnan(env_data).any():
                    valid_mask = ~torch.isnan(env_data)
                    if valid_mask.any():
                        # NaN이 아닌 값들의 평균 계산
                        mean_val = torch.mean(env_data[valid_mask])
                        data[env_idx] = torch.where(torch.isnan(env_data), mean_val, env_data)
                    else:
                        # 모든 값이 NaN인 경우 기본값 사용
                        data[env_idx] = torch.where(torch.isnan(env_data), 
                                                   torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                                                   env_data)
        else:
            # 1D 텐서의 경우 전체 평균 계산
            valid_mask = ~torch.isnan(data)
            if valid_mask.any():
                mean_val = torch.mean(data[valid_mask])
                data = torch.where(torch.isnan(data), mean_val, data)
            else:
                data = torch.where(torch.isnan(data), 
                                 torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                                 data)
    elif method == "previous":
        # 이전 값으로 대체 (forward fill)
        if data.dim() == 1:
            # 1D 텐서의 경우
            for i in range(1, data.shape[0]):
                if torch.isnan(data[i]):
                    data[i] = data[i-1]
            # 첫 번째 값이 NaN인 경우 기본값 사용
            if torch.isnan(data[0]):
                data[0] = replacement_value
        elif data.dim() == 2:
            # 2D 텐서 (num_envs, features)의 경우 각 환경별로 처리
            # 벡터화된 연산으로 개선
            for env_idx in range(data.shape[0]):
                env_data = data[env_idx]
                if torch.isnan(env_data).any():
                    # forward fill: 이전 값으로 NaN 대체
                    nan_mask = torch.isnan(env_data)
                    if nan_mask.any():
                        # 첫 번째 값이 NaN인 경우 기본값으로 설정
                        if nan_mask[0]:
                            env_data[0] = replacement_value
                        
                        # forward fill 적용
                        for i in range(1, env_data.shape[0]):
                            if nan_mask[i]:
                                env_data[i] = env_data[i-1]
                        
                        data[env_idx] = env_data
        else:
            # 다차원 텐서의 경우 기본값 사용
            data = torch.where(torch.isnan(data), 
                             torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                             data)
    else:
        # 알 수 없는 방법인 경우 기본값 사용
        data = torch.where(torch.isnan(data), 
                         torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                         data)
    
    # 최종 검증: 여전히 NaN이 있는지 확인
    if torch.isnan(data).any():
        if enable_logging:
            print(f"[NaN Safety] {obs_name}: WARNING: NaN values still present after replacement!")
        # 최후의 수단: 모든 NaN을 기본값으로 대체
        data = torch.where(torch.isnan(data), 
                         torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                         data)
    
    # 추가 검증: 무한대 값이나 매우 큰 값 처리
    if torch.isinf(data).any() or (data.abs() > 1e6).any():
        if enable_logging:
            print(f"[NaN Safety] {obs_name}: WARNING: Infinite or very large values detected!")
        # 무한대 값과 매우 큰 값을 기본값으로 대체
        data = torch.where(torch.isinf(data) | (data.abs() > 1e6), 
                         torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                         data)
    
    # 최종 검증: 모든 값이 유효한 범위 내에 있는지 확인
    if torch.isnan(data).any() or torch.isinf(data).any() or (data.abs() > 1e6).any():
        if enable_logging:
            print(f"[NaN Safety] {obs_name}: CRITICAL: Final validation failed! Forcing replacement.")
        # 모든 문제가 있는 값을 기본값으로 대체
        data = torch.where(torch.isnan(data) | torch.isinf(data) | (data.abs() > 1e6), 
                         torch.tensor(replacement_value, device=data.device, dtype=data.dtype), 
                         data)
    
    return data


def get_environment_groups(num_envs: int, num_groups: int = 4):
    """환경들을 그룹으로 나누는 유틸리티 함수.
    
    Args:
        num_envs: 전체 환경 수
        num_groups: 그룹 수 (기본값: 4)
        
    Returns:
        각 그룹의 환경 ID 리스트
    """
    envs_per_group = num_envs // num_groups
    remainder = num_envs % num_groups
    
    groups = []
    start_idx = 0
    
    for i in range(num_groups):
        # 마지막 그룹에 나머지 환경들을 모두 할당
        if i == num_groups - 1:
            end_idx = num_envs
        else:
            end_idx = start_idx + envs_per_group + (1 if i < remainder else 0)
        
        groups.append(torch.arange(start_idx, end_idx))
        start_idx = end_idx
    
    return groups


def configure_event_groups(env_cfg, num_envs: int):
    """이벤트 그룹을 환경 설정에 적용하는 함수.
    
    Args:
        env_cfg: 환경 설정 객체
        num_envs: 전체 환경 수
    """
    groups = get_environment_groups(num_envs, 4)
    
    # 각 이벤트를 해당 그룹에만 적용 (Tensor를 리스트로 변환)
    env_cfg.events.push_robot.env_ids = groups[0].tolist()
    env_cfg.events.specific_velocity_push.env_ids = groups[1].tolist()
    env_cfg.events.external_wrench_with_offset.env_ids = groups[2].tolist()
    env_cfg.events.specific_external_wrench.env_ids = groups[3].tolist()



# USD path: defaults to local .usda; can be overridden by environment variable.
DEFAULT_SUITECASE_USD = "/home/eric/sequor_robotics/sequor_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/suitecase/assets/Auto_Balancing_Case.usda"

# Single balancing joint name
BALANCE_JOINT_NAME = "Revolute_motor"
# Root Base
ROOT_BASE_NAME = "base_link"
# Wheel bodies regex for contact tracking
WHEEL_BODIES_REGEX = ".*_wheel_1"
# External wrench application body (handle). In USD, `handle` is not a rigid body; default to `luggage_case_1`.
HANDLE_BODY_NAME = "luggage_case_1"
CASE_BODY_NAME = "luggage_case_1"
# Observation history length (joint state/action)
OBS_HISTORY_LENGTH = 4

# =============================================================================
# USD Asset Body Structure (Auto_Balancing_Case.usd)
# =============================================================================
# Available Rigid Bodies
# - Root: "base_link" (main platform, mass: ~0.98kg)
# - "FR_wheelbase_1", "RR_wheelbase_1", "FL_wheelbase_1", "RL_wheelbase_1" (wheel mounts)
# - "FR_wheel_1", "RR_wheel_1", "FL_wheel_1", "RL_wheel_1" (wheels, mass: ~0.042kg each)
# - "luggage_case_1" (main case body, mass: ~5.6kg, contains handle as sub-Xform)
#
# Joint Structure:
# - "Revolute_motor" (Y-axis, -30° to +30° limits) - base_link ↔ luggage_case_1
# - "Revolute_1" to "Revolute_8" - wheelbase and wheel joints
#
# Note: "handle" is a sub-Xform under "luggage_case_1", not a separate rigid body
# =============================================================================




SUITECASE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEFAULT_SUITECASE_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "balance_hinge": ImplicitActuatorCfg(
            joint_names_expr=[BALANCE_JOINT_NAME],
            stiffness=10.0,
            damping=1.0,
            effort_limit_sim=3.0,
            velocity_limit_sim=6,
        )
    },
)


@configclass
class SuitecaseSceneCfg(InteractiveSceneCfg):
    """Simple planar scene for the suitcase robot."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    # Robot
    robot: ArticulationCfg = SUITECASE_CFG

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )

    # Sensors
    wheel_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Auto_Balancing_Case/" + WHEEL_BODIES_REGEX,
        history_length=3,
        track_air_time=True,
    )
    
    handle_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Auto_Balancing_Case/" + HANDLE_BODY_NAME,
        history_length=3,
        track_air_time=True,
    )


@configclass
class ActionsCfg:
    """MDP action: single hinge (balancing) joint target position (1D)."""

    # Single joint position control (set SUITECASE_BALANCE_JOINT_NAME to match your joint name if different)
    hinge_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[BALANCE_JOINT_NAME],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
        clip={BALANCE_JOINT_NAME: (-0.5, 0.5)},  # less than -30° to +30° in radians
    )


@configclass
class ObservationsCfg:
    """MDP observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy input."""

        # Balancing hinge joint state (relative position and velocity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME])},
            noise=Unoise(n_min=-0.005, n_max=0.005),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "zero", "obs_name": "joint_pos"})],
            history_length=OBS_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "zero", "obs_name": "joint_vel"})],
            history_length=OBS_HISTORY_LENGTH,
        )

        # Previous action (for action smoothing/history)
        prev_action = ObsTerm(
            func=mdp.last_action, 
            params={"action_name": "hinge_pos"}, 
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "previous", "obs_name": "prev_action"})],
            history_length=OBS_HISTORY_LENGTH
        )

        # Contact force magnitude for all 4 wheels
        wheel_contact_forces = ObsTerm(
            func=wheel_contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=WHEEL_BODIES_REGEX)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "mean", "obs_name": "wheel_contact_forces"})],
            history_length=OBS_HISTORY_LENGTH,
        )
        
        # Contact force magnitude for handle
        handle_contact_forces = ObsTerm(
            func=handle_contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("handle_contact_forces", body_names=[HANDLE_BODY_NAME])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "mean", "obs_name": "handle_contact_forces"})],
            history_length=OBS_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Optional asymmetric critic observations (can include ground truth)."""

        # Include same signals as policy
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME])},
            noise=Unoise(n_min=-0.005, n_max=0.005),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "zero", "obs_name": "critic_joint_pos"})],
            history_length=OBS_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME])},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "zero", "obs_name": "critic_joint_vel"})],
            history_length=OBS_HISTORY_LENGTH,
        )
        prev_action = ObsTerm(
            func=mdp.last_action, 
            params={"action_name": "hinge_pos"}, 
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "previous", "obs_name": "critic_prev_action"})],
            history_length=OBS_HISTORY_LENGTH
        )

        # Contact force magnitude for all 4 wheels
        wheel_contact_forces = ObsTerm(
            func=wheel_contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=WHEEL_BODIES_REGEX)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "mean", "obs_name": "critic_wheel_contact_forces"})],
            history_length=OBS_HISTORY_LENGTH,
        )
        
        # Contact force magnitude for handle
        handle_contact_forces = ObsTerm(
            func=handle_contact_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("handle_contact_forces", body_names=[HANDLE_BODY_NAME])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            modifiers=[ModifierCfg(func=safe_nan_replacement, params={"replacement_value": 0.0, "method": "mean", "obs_name": "critic_handle_contact_forces"})],
            history_length=OBS_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Use asymmetric critic if needed
    critic: ObsGroup = CriticCfg()
    

@configclass
class CommandsCfg:
    """No explicit commands (balancing-only task)."""
    pass


@configclass
class EventCfg:
    """Randomization/reset events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.0, 0.0),
        },
    )

    
    # Push the robot by setting velocity for robustness
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0),    # Linear velocity push
                "y": (-1.0, 1.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),  # Angular velocity push
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }, 
    )
    
    # Specific velocity push applied to robot (targeted disturbance)
    specific_velocity_push = EventTerm(
        func=push_by_setting_specific_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "vel_x_range": (-2.0, 2.0),  # x-direction velocity range
            "vel_y": 0.0,  # y-direction velocity (fixed)
            "vel_z": 0.0,  # z-direction velocity (fixed)
            "ang_vel_x": 0.0,  # roll (fixed)
            "ang_vel_y": 0.0,  # pitch (fixed)
            "ang_vel_z": 0.0,  # yaw (fixed)
        },
    )
    
    
    # External force/torque disturbance with position offset applied on handle/body
    external_wrench_with_offset = EventTerm(
        func=apply_external_force_torque_offset,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[HANDLE_BODY_NAME]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-1.0, 1.0),
            "position_offset": (0.0, 0.0, 0.89),  # Offset from body center of mass (x, y, z)
        },
    )
    
    # Specific external force/torque applied on handle/body (example)
    specific_external_wrench = EventTerm(
        func=apply_specific_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[HANDLE_BODY_NAME]),
            "force_x_range": (-10.0, 10.0),  # x-direction force range
            "force_y": 0.0,  # y-direction force (fixed)
            "force_z": 0.0,  # z-direction force (fixed)
            "torque_x_range": (-5.0, 5.0),  # x-direction torque range
            "torque_y": 0.0,  # y-direction torque (fixed)
            "torque_z": 0.0,  # z-direction torque (fixed)
            "position_offset": (0.0, 0.0, 0.89),  # Adjusted offset
        },
    )
    
    '''
    # Randomize handle body mass for robustness
    randomize_handle_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[CASE_BODY_NAME]),
            "mass_distribution_params": (0.8, 1.2),  # ±20% mass variation
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    
    # Randomize handle center of mass for robustness
    randomize_handle_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[CASE_BODY_NAME]),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )
    
    # Randomize actuator gains for robustness
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME]),
            "stiffness_distribution_params": (0.9, 1.1),  # ±30% stiffness variation
            "damping_distribution_params": (0.9, 1.1),    # ±30% damping variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    # Randomize joint parameters for robustness
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME]),
            "friction_distribution_params": (0.001, 0.01),  # Joint friction variation
            "armature_distribution_params": (0.001, 0.005), # Joint armature variation
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    '''



@configclass
class RewardsCfg:
    """Reward functions: balancing task."""

    # (0) Surviving reward - 로봇이 살아있을 때마다 보상
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,  # reward for staying alive
    )

    # (-1) Termination penalty - 로봇이 죽을 때 penalty
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-50.0,  # large penalty for termination
    )

    # (1) ROOT_BASE_NAME이 수평을 최대한 유지하도록 - 기본 평면 orientation 유지
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-10.0,  # penalty for non-flat orientation
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # (2) ROOT_BASE_NAME의 angular velocity 감소 (수평 유지 도움)
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.5,  # penalty for angular motion
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # (3) BALANCE_JOINT_NAME이 default position과 비슷하도록
    hinge_pos_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,  # penalty for deviation from default
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[BALANCE_JOINT_NAME])},
    )
    
    # (4) 4개의 접촉이 떨어지지 않도록 - wheel contact 유지
    desired_contacts = RewTerm(
        func=mdp.desired_contacts,
        weight=-10.0,  # penalty when no contact
        params={
            "sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=WHEEL_BODIES_REGEX),
            "threshold": 1.0
        },
    )
    
    # (5) 4개 wheel의 contact force 값이 최대한 비슷하도록 (variance 기반)
    wheel_contact_balance = RewTerm(
        func=wheel_contact_force_variance,
        weight=1.0,  # reward for balanced contact forces
        params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=WHEEL_BODIES_REGEX)},
    )
    
    # (6) 4개 wheel의 contact force 최대-최소 차이 penalize
    wheel_contact_min_max_penalty = RewTerm(
        func=wheel_contact_force_min_max,
        weight=-0.1,  # penalty for unbalanced contact forces
        params={"sensor_cfg": SceneEntityCfg("wheel_contact_forces", body_names=WHEEL_BODIES_REGEX)},
    )

    # (7) Action smoothness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)



@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Terminate on large tilt
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    
    # Terminate on NaN detection in robot states
    nan_detection = DoneTerm(
        func=nan_detection_termination,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # Terminate on extreme values that indicate simulation instability
    extreme_values = DoneTerm(
        func=extreme_values_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_limit": 50.0,
            "angular_velocity_limit": 50.0
        }
    )
    

@configclass
class SuitecaseEnvCfg(ManagerBasedRLEnvCfg):
    """Suitcase robot balancing environment configuration."""

    # 장면/MDP
    scene: SuitecaseSceneCfg = SuitecaseSceneCfg(num_envs=2048, env_spacing=3.0,)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Simulation/general settings
        self.decimation = 4  # 50 Hz control (dt=0.005 * 4)
        self.episode_length_s = 8.0
        self.sim.dt = 0.005  # 200 Hz physics
        self.sim.render_interval = self.decimation
        # Physics material settings
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        # Sensors update period
        if getattr(self.scene, "wheel_contact_forces", None) is not None:
            self.scene.wheel_contact_forces.update_period = self.sim.dt
        if getattr(self.scene, "handle_contact_forces", None) is not None:
            self.scene.handle_contact_forces.update_period = self.sim.dt
        
        # Configure event groups to avoid overlap
        configure_event_groups(self, self.scene.num_envs)


@configclass
class SuitecasePlayEnvCfg(SuitecaseEnvCfg):
    """Suitcase robot balancing environment configuration for PLAY/Inference."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        self.observations.policy.enable_corruption = False
        
        # Configure event groups for play environment
        configure_event_groups(self, self.scene.num_envs)