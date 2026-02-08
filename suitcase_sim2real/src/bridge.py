#!/usr/bin/env python3
"""
Auto Balancing Case Sim2Real Bridge

Bridges the Isaac Lab-trained RL policy to the real Auto Balancing Case hardware,
orchestrating a 50Hz real-time control loop with Dynamixel motors and HX711 load cells.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List
from collections import deque

from motor_interface import DynamixelXL430Interface
from sensor_interface import HX711LoadCellInterface
from policy_interface import PolicyInterface, PolicyConfig
from config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AutoBalancingCaseBridge:
    """Sim2Real bridge for the Auto Balancing Case.

    Connects the trained PPO policy to real hardware, managing observation
    construction (4-step history x 8 dims = 32-dim input), motor commands
    (4 motors across 2 ports), and sensor readings (5 load cells via Arduino).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the bridge with configuration and hardware interfaces.

        Args:
            config_path: Path to the YAML configuration file.
                         Defaults to config/interface_config.yml.
        """
        self.config_manager = ConfigManager(config_path)

        if not self.config_manager.validate_config():
            raise ValueError("Configuration validation failed")

        # Store per-interface config references
        self.policy_config = self.config_manager.policy
        self.sensor_config = self.config_manager.sensor
        self.actuator_config = self.config_manager.actuator
        self.system_config = self.config_manager.system

        # Initialize policy interface
        policy_config = PolicyConfig(
            model_path=self.policy_config.model_path,
            device=self.policy_config.device,
            obs_history_length=self.policy_config.obs_history_length,
            actor_hidden_dims=self.policy_config.actor_hidden_dims,
            critic_hidden_dims=self.policy_config.critic_hidden_dims,
            activation=self.policy_config.activation,
            init_noise_std=self.policy_config.init_noise_std,
            noise_std_type=self.policy_config.noise_std_type,
            obs_dim_per_timestep=self.policy_config.obs_dim_per_timestep,
            num_actions=self.policy_config.num_actions
        )
        self.policy_interface = PolicyInterface(policy_config)

        # Initialize hardware
        self._initialize_hardware()

        # Observation and action history (bounded deques)
        self.obs_history: deque = deque(maxlen=self.policy_config.obs_history_length)
        self.action_history: deque = deque(maxlen=self.policy_config.obs_history_length)

        # Control timing
        self.control_dt: float = 1.0 / self.policy_config.control_frequency

        # Runtime state
        self.episode_step: int = 0
        self.is_running: bool = False
        self.emergency_stop: bool = False
        self.initial_joint_pos: float = 0.0  # For Isaac Lab relative position computation

        logger.info("Auto Balancing Case Sim2Real Bridge initialized")
        logger.info("Config Summary: %s", self.config_manager.get_config_summary())
        logger.info("Policy Info: %s", self.policy_interface.get_model_info())

    def _initialize_hardware(self) -> None:
        """Initialize motor and sensor hardware interfaces."""
        # Dynamixel quad-motor interface (2 ports)
        self.motor_interface = DynamixelXL430Interface(
            motor_ids=self.actuator_config.all_motor_ids,
            device_names=self.actuator_config.devices,
            baudrate=self.actuator_config.baudrate,
            control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
        )

        # Arduino-based load cell interface
        self.load_cell_interface = HX711LoadCellInterface(
            arduino_port=self.sensor_config.port,
            baudrate=self.sensor_config.baudrate
        )

        # Attempt to load calibration data
        try:
            self.load_cell_interface.load_calibration()
            logger.info("Load cell calibration data loaded")
        except Exception as e:
            logger.warning("No load cell calibration data found: %s. Run calibration first.", e)

    def _get_current_observation(self) -> Dict[str, np.ndarray]:
        """Read current hardware state and convert to policy observation format.

        Constructs an 8-dimensional observation vector matching the Isaac Lab
        environment: [joint_pos, joint_vel, prev_action, wheel_forces(4), handle_force].

        Returns:
            Dictionary with observation components as numpy arrays.
        """
        # Motor state (already in radians from the interface)
        motor_state = self.motor_interface.get_state()
        joint_pos_rad: float = motor_state['position']
        joint_vel_rad_s: float = motor_state['velocity']

        # Load cell state
        load_cell_state = self.load_cell_interface.get_state()
        wheel_forces: np.ndarray = load_cell_state['wheel_forces']  # [FR, RR, FL, RL]
        handle_force: float = load_cell_state['handle_force'][0]

        # Previous action (default to 0.0 if no history)
        prev_action = self.action_history[-1] if len(self.action_history) > 0 else 0.0

        # Compute relative joint position (same as Isaac Lab's joint_pos_rel)
        relative_joint_pos = joint_pos_rad - self.initial_joint_pos

        return {
            'joint_pos': np.array([relative_joint_pos]),
            'joint_vel': np.array([joint_vel_rad_s]),
            'prev_action': np.array([prev_action]),
            'wheel_contact_forces': wheel_forces,
            'handle_external_force': np.array([handle_force])
        }  # 8 dims per timestep

    def _log_step_info(self, step: int, raw_obs: Dict[str, np.ndarray],
                       abs_joint_pos: float, raw_action: float, action: float) -> None:
        """Log detailed step information for debugging.

        Args:
            step: Current step number.
            raw_obs: Raw observation dictionary.
            abs_joint_pos: Absolute joint position in radians.
            raw_action: Unclipped policy output.
            action: Clipped action value.
        """
        rel_pos = raw_obs['joint_pos'][0]
        vel = raw_obs['joint_vel'][0]
        prev_act = raw_obs['prev_action'][0]
        wf = raw_obs['wheel_contact_forces']
        hf = raw_obs['handle_external_force'][0]
        target = action

        logger.info(
            "Step %4d:\n"
            "  [POLICY INPUT] JointPos=%+.3frad(%+.1f deg) JointVel=%+.3frad/s PrevAction=%+.3f\n"
            "  [POLICY INPUT] WheelForces=[%.1f,%.1f,%.1f,%.1f]N HandleForce=%.1fN\n"
            "  [POLICY OUTPUT] RawAction=%+.3f ClippedAction=%+.3f TargetAngle=%+.3frad(%+.1f deg)\n"
            "  [HARDWARE STATE] AbsJointPos=%+.3frad(%+.1f deg)",
            step,
            rel_pos, np.degrees(rel_pos), vel, prev_act,
            wf[0], wf[1], wf[2], wf[3], hf,
            raw_action, action, target, np.degrees(target),
            abs_joint_pos, np.degrees(abs_joint_pos)
        )

    def _run_control_loop(self, max_steps: Optional[int] = None, log_interval: int = 25) -> None:
        """Core control loop shared by episode and continuous modes.

        Executes the full sim2real pipeline: hardware init -> observation history
        warmup -> real-time control loop -> safe shutdown.

        Args:
            max_steps: Maximum number of control steps. None for unlimited (continuous mode).
            log_interval: Print debug info every N steps.
        """
        mode_name = "episode" if max_steps is not None else "continuous"
        logger.info("Starting %s control...", mode_name)

        # Start hardware reading threads
        self.motor_interface.start_real_time_reading()
        self.load_cell_interface.start_real_time_reading()

        # Move to center position and wait for settling
        logger.info("Moving to center position...")
        self.motor_interface.set_command(0.0)
        time.sleep(3.0)

        # Record initial position for relative position computation
        motor_state = self.motor_interface.get_state()
        self.initial_joint_pos = motor_state['position']
        logger.info("Initial joint position: %.3f rad", self.initial_joint_pos)

        # Reset state
        step_count = 0
        self.is_running = True
        self.emergency_stop = False
        self.obs_history.clear()
        self.action_history.clear()

        # Warm up observation history with real sensor data (no normalization)
        logger.info("Initializing observation history...")
        for _ in range(self.policy_config.obs_history_length):
            obs = self._get_current_observation()
            self.obs_history.append(obs)
            self.action_history.append(0.0)
            time.sleep(self.control_dt)

        logger.info("Control loop started!")

        try:
            while self.is_running and not self.emergency_stop:
                step_start_time = time.time()

                # 1. Read current sensor state (raw values, no normalization)
                raw_obs = self._get_current_observation()

                # 2. Safety check (absolute position)
                motor_state = self.motor_interface.get_state()
                current_joint_pos_abs = motor_state['position']
                if abs(current_joint_pos_abs) > self.system_config.emergency_angle_limit:
                    logger.error(
                        "Emergency stop: Joint angle %.3f rad exceeds limit %.3f rad",
                        current_joint_pos_abs, self.system_config.emergency_angle_limit
                    )
                    self.emergency_stop = True
                    break

                # 3. Check episode length limit (episode mode only)
                if max_steps is not None and step_count >= max_steps:
                    logger.info("Episode finished: max steps %d reached", max_steps)
                    break

                # 4. Update observation history
                self.obs_history.append(raw_obs)

                # 5. Run policy inference
                try:
                    obs_history_list = list(self.obs_history)
                    raw_action = self.policy_interface.predict(obs_history_list)
                    action = float(np.clip(raw_action, -0.5, 0.5))
                except Exception as e:
                    logger.error("Policy prediction error: %s", e)
                    action = 0.0
                    raw_action = 0.0

                # 6. Update action history
                self.action_history.append(action)

                # 7. Send motor command (action is already in radians [-0.5, 0.5])
                self.motor_interface.set_command(action)

                # 8. Periodic debug logging
                if step_count % log_interval == 0:
                    self._log_step_info(step_count, raw_obs, current_joint_pos_abs,
                                        raw_action, action)

                # 9. Maintain control frequency
                elapsed = time.time() - step_start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:
                    logger.warning("Control loop delay: %.3fs", -sleep_time)

                step_count += 1

        except KeyboardInterrupt:
            logger.info("Control stopped by user")

        except Exception as e:
            logger.error("Control loop error: %s", e)
            self.emergency_stop = True

        finally:
            logger.info("Shutting down...")
            self.motor_interface.set_command(0.0)
            time.sleep(2.0)
            self.motor_interface.stop_real_time_reading()
            self.load_cell_interface.stop_real_time_reading()
            logger.info("Control loop finished after %d steps", step_count)

    def run_episode(self) -> None:
        """Run a single episode (limited to max_episode_steps from config)."""
        self._run_control_loop(
            max_steps=self.system_config.max_episode_steps,
            log_interval=25  # Every 0.5s at 50Hz
        )

    def run_continuous(self) -> None:
        """Run continuously until interrupted (Ctrl+C)."""
        self._run_control_loop(
            max_steps=None,
            log_interval=50  # Every 1.0s at 50Hz
        )

    def calibrate_load_cells(self) -> None:
        """Run load cell calibration procedure using a reference weight."""
        logger.info("Starting load cell calibration...")
        self.load_cell_interface.calibrate_all_load_cells(200.0)
        self.load_cell_interface.save_calibration()

    def shutdown(self) -> None:
        """Safely shut down all hardware interfaces."""
        self.is_running = False
        self.motor_interface.shutdown()
        self.load_cell_interface.shutdown()
        logger.info("Auto Balancing Case Bridge shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    bridge = AutoBalancingCaseBridge()
    try:
        bridge.run_episode()
    finally:
        bridge.shutdown()
