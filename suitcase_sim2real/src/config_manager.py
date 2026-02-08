#!/usr/bin/env python3
"""
Configuration Manager for Auto Balancing Case

Loads interface_config.yml, applies platform-specific overrides (Windows/Linux),
and provides typed dataclass objects for each subsystem (policy, sensor, actuator, system).
"""

import yaml
import os
import platform
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PolicyInterfaceConfig:
    """Policy network and inference configuration."""
    model_path: str
    device: str = "cpu"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    obs_history_length: int = 4
    obs_dim_per_timestep: int = 8       # joint_pos(1) + joint_vel(1) + prev_action(1) + wheel_forces(4) + handle_force(1)
    num_actions: int = 1                # Single balance joint
    control_frequency: float = 50.0     # Hz (matches Isaac Lab decimation=4, dt=0.005)


@dataclass
class SensorInterfaceConfig:
    """HX711 load cell sensor configuration."""
    # Serial connection
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    timeout: float = 2.0

    # Calibration
    auto_load_on_startup: bool = True
    calibration_file: str = "load_cell_calibration.npz"
    reference_weight: float = 200.0     # grams

    # Sensor layout
    wheel_sensor_count: int = 4
    wheel_sensor_names: List[str] = field(default_factory=lambda: ["FL", "FR", "RL", "RR"])
    wheel_channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    handle_sensor_count: int = 1
    handle_sensor_names: List[str] = field(default_factory=lambda: ["handle"])
    handle_channels: List[int] = field(default_factory=lambda: [4])

    # Signal processing
    sampling_rate: float = 100.0        # Hz
    filter_enabled: bool = True
    filter_type: str = "moving_average"
    filter_window: int = 5
    noise_threshold: float = 0.1        # N


@dataclass
class ActuatorInterfaceConfig:
    """Dynamixel XL430 motor configuration (2 ports, 4 motors)."""
    # Serial connection (2 ports)
    devices: List[str] = field(default_factory=lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1"])
    baudrate: int = 57600
    protocol_version: float = 2.0
    timeout: float = 1.0

    # Motor grouping: port1 = [1,2] (+theta), port2 = [3,4] (-theta)
    port1_motor_ids: List[int] = field(default_factory=lambda: [1, 2])
    port2_motor_ids: List[int] = field(default_factory=lambda: [3, 4])
    all_motor_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    control_mode: str = "position"
    profile_velocity: int = 50          # 0-1023 (0 = max speed)
    profile_acceleration: int = 20      # 0-32767

    # Safety limits
    position_limit_min: float = -0.5    # rad
    position_limit_max: float = 0.5     # rad
    velocity_limit: float = 6.0         # rad/s
    current_limit: int = 500            # mA
    temperature_limit: int = 80         # Celsius

    # PID gains
    position_p: int = 800
    position_i: int = 0
    position_d: int = 0

    # Monitoring
    monitoring_enabled: bool = True
    monitoring_frequency: float = 100.0  # Hz
    log_data: bool = False


@dataclass
class SystemConfig:
    """Global system configuration."""
    # Safety
    emergency_stop_enabled: bool = True
    emergency_angle_limit: float = 0.51     # rad (emergency stop threshold)
    emergency_force_limit: float = 100.0    # N
    watchdog_enabled: bool = True
    watchdog_timeout: float = 1.0           # seconds

    # Episode
    max_episode_steps: int = 4000           # 50Hz * 8s (matches Isaac Lab)
    episode_timeout: float = 8.0            # seconds
    auto_reset: bool = True

    # Logging
    logging_enabled: bool = True
    log_level: str = "INFO"
    log_file: str = "auto_balancing_case.log"
    console_output: bool = True
    data_logging_enabled: bool = False
    data_logging_frequency: float = 50.0    # Hz
    data_output_dir: str = "logs/data"
    data_format: str = "csv"

    # Performance
    monitor_timing: bool = True
    target_control_frequency: float = 50.0  # Hz
    timing_warning_threshold: float = 0.02  # seconds (20ms)
    timing_error_threshold: float = 0.05    # seconds (50ms)


class ConfigManager:
    """YAML configuration loader with platform-specific overrides.

    Loads interface_config.yml, detects the current platform (Windows/Linux),
    applies serial port overrides, and creates typed dataclass config objects
    for each subsystem.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize and load configuration.

        Args:
            config_path: Path to YAML config file. Defaults to config/interface_config.yml.
        """
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = str(current_dir / "config" / "interface_config.yml")

        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self._apply_platform_overrides()

        # Create typed config objects
        self.policy = self._create_policy_config()
        self.sensor = self._create_sensor_config()
        self.actuator = self._create_actuator_config()
        self.system = self._create_system_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file.

        Returns:
            Raw configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _apply_platform_overrides(self) -> None:
        """Apply platform-specific configuration overrides (e.g., serial ports)."""
        current_platform = platform.system().lower()

        if 'platform_overrides' not in self.raw_config:
            return

        platform_config = self.raw_config['platform_overrides'].get(current_platform, {})
        for key_path, value in platform_config.items():
            self._set_nested_value(self.raw_config, key_path, value)

    @staticmethod
    def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot-separated key path.

        Args:
            config: Root configuration dictionary.
            key_path: Dot-separated path (e.g., 'sensor.load_cell.connection.port').
            value: Value to set.
        """
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _create_policy_config(self) -> PolicyInterfaceConfig:
        """Build PolicyInterfaceConfig from raw YAML data."""
        cfg = self.raw_config.get('policy', {})
        return PolicyInterfaceConfig(
            model_path=cfg.get('model_path', ''),
            device=cfg.get('device', 'cpu'),
            actor_hidden_dims=cfg.get('actor_hidden_dims', [256, 128, 64]),
            critic_hidden_dims=cfg.get('critic_hidden_dims', [256, 128, 64]),
            activation=cfg.get('activation', 'elu'),
            init_noise_std=cfg.get('init_noise_std', 1.0),
            noise_std_type=cfg.get('noise_std_type', 'scalar'),
            obs_history_length=cfg.get('obs_history_length', 4),
            obs_dim_per_timestep=cfg.get('obs_dim_per_timestep', 8),
            num_actions=cfg.get('num_actions', 1),
            control_frequency=cfg.get('control_frequency', 50.0),
        )

    def _create_sensor_config(self) -> SensorInterfaceConfig:
        """Build SensorInterfaceConfig from raw YAML data."""
        sensor_cfg = self.raw_config.get('sensor', {})
        lc = sensor_cfg.get('load_cell', {})
        conn = lc.get('connection', {})
        calib = lc.get('calibration', {})
        sensors = lc.get('sensors', {})
        wheel = sensors.get('wheel_sensors', {})
        handle = sensors.get('handle_sensor', {})
        proc = lc.get('processing', {})

        return SensorInterfaceConfig(
            port=conn.get('port', '/dev/ttyACM0'),
            baudrate=conn.get('baudrate', 115200),
            timeout=conn.get('timeout', 2.0),
            auto_load_on_startup=calib.get('auto_load_on_startup', True),
            calibration_file=calib.get('calibration_file', 'load_cell_calibration.npz'),
            reference_weight=calib.get('reference_weight', 200.0),
            wheel_sensor_count=wheel.get('count', 4),
            wheel_sensor_names=wheel.get('names', ['FL', 'FR', 'RL', 'RR']),
            wheel_channels=wheel.get('channels', [0, 1, 2, 3]),
            handle_sensor_count=handle.get('count', 1),
            handle_sensor_names=handle.get('names', ['handle']),
            handle_channels=handle.get('channels', [4]),
            sampling_rate=proc.get('sampling_rate', 100.0),
            filter_enabled=proc.get('filter_enabled', True),
            filter_type=proc.get('filter_type', 'moving_average'),
            filter_window=proc.get('filter_window', 5),
            noise_threshold=proc.get('noise_threshold', 0.1),
        )

    def _create_actuator_config(self) -> ActuatorInterfaceConfig:
        """Build ActuatorInterfaceConfig from raw YAML data."""
        act = self.raw_config.get('actuator', {})
        motor = act.get('motor', {})
        conn = motor.get('connection', {})
        motors = motor.get('motors', {})
        p1 = motors.get('port1_motors', {})
        p2 = motors.get('port2_motors', {})
        # Backward compatibility with older config format
        legacy = motors.get('balancing_joint', {})
        limits = motor.get('limits', {})
        ctrl = motor.get('control', {})
        pid = ctrl.get('pid_gains', {})
        mon = ctrl.get('monitoring', {})

        port1_ids = p1.get('ids', legacy.get('ids', [1, 2])[:2])
        port2_ids = p2.get('ids', [3, 4])

        return ActuatorInterfaceConfig(
            devices=conn.get('devices', ['/dev/ttyUSB0', '/dev/ttyUSB1']),
            baudrate=conn.get('baudrate', 57600),
            protocol_version=conn.get('protocol_version', 2.0),
            timeout=conn.get('timeout', 1.0),
            port1_motor_ids=port1_ids,
            port2_motor_ids=port2_ids,
            all_motor_ids=port1_ids + port2_ids,
            control_mode=p1.get('control_mode', legacy.get('control_mode', 'position')),
            profile_velocity=p1.get('profile_velocity', legacy.get('profile_velocity', 50)),
            profile_acceleration=p1.get('profile_acceleration', legacy.get('profile_acceleration', 20)),
            position_limit_min=limits.get('position_limit', {}).get('min', -0.5),
            position_limit_max=limits.get('position_limit', {}).get('max', 0.5),
            velocity_limit=limits.get('velocity_limit', 6.0),
            current_limit=limits.get('current_limit', 500),
            temperature_limit=limits.get('temperature_limit', 80),
            position_p=pid.get('position_p', 800),
            position_i=pid.get('position_i', 0),
            position_d=pid.get('position_d', 0),
            monitoring_enabled=mon.get('enabled', True),
            monitoring_frequency=mon.get('frequency', 100.0),
            log_data=mon.get('log_data', False),
        )

    def _create_system_config(self) -> SystemConfig:
        """Build SystemConfig from raw YAML data."""
        sys_cfg = self.raw_config.get('system', {})
        safety = sys_cfg.get('safety', {})
        estop = safety.get('emergency_stop', {})
        wd = safety.get('watchdog', {})
        session = sys_cfg.get('session', {})
        log = sys_cfg.get('logging', {})
        data_log = log.get('data_logging', {})
        perf = sys_cfg.get('performance', {})

        return SystemConfig(
            emergency_stop_enabled=estop.get('enabled', True),
            emergency_angle_limit=estop.get('angle_limit', 0.51),
            emergency_force_limit=estop.get('force_limit', 100.0),
            watchdog_enabled=wd.get('enabled', True),
            watchdog_timeout=wd.get('timeout', 1.0),
            max_episode_steps=session.get('max_episode_steps', 4000),
            episode_timeout=session.get('episode_timeout', 8.0),
            auto_reset=session.get('auto_reset', True),
            logging_enabled=log.get('enabled', True),
            log_level=log.get('level', 'INFO'),
            log_file=log.get('log_file', 'auto_balancing_case.log'),
            console_output=log.get('console_output', True),
            data_logging_enabled=data_log.get('enabled', False),
            data_logging_frequency=data_log.get('frequency', 50.0),
            data_output_dir=data_log.get('output_dir', 'logs/data'),
            data_format=data_log.get('format', 'csv'),
            monitor_timing=perf.get('monitor_timing', True),
            target_control_frequency=perf.get('target_control_frequency', 50.0),
            timing_warning_threshold=perf.get('timing_warning_threshold', 0.02),
            timing_error_threshold=perf.get('timing_error_threshold', 0.05),
        )

    def get_config_summary(self) -> Dict[str, Any]:
        """Return a summary of the current configuration for logging."""
        return {
            "config_file": str(self.config_path),
            "platform": platform.system(),
            "policy": {
                "model_path": self.policy.model_path,
                "device": self.policy.device,
                "control_frequency": self.policy.control_frequency,
            },
            "sensor": {
                "port": self.sensor.port,
                "baudrate": self.sensor.baudrate,
            },
            "actuator": {
                "devices": self.actuator.devices,
                "port1_motor_ids": self.actuator.port1_motor_ids,
                "port2_motor_ids": self.actuator.port2_motor_ids,
            },
            "system": {
                "max_episode_steps": self.system.max_episode_steps,
                "emergency_angle_limit": self.system.emergency_angle_limit,
            },
        }

    def validate_config(self) -> bool:
        """Validate configuration values.

        Returns:
            True if all validations pass.
        """
        errors: List[str] = []

        if not os.path.exists(self.policy.model_path):
            errors.append(f"Policy model file not found: {self.policy.model_path}")

        if self.policy.control_frequency <= 0:
            errors.append(f"Invalid control frequency: {self.policy.control_frequency}")

        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error("  - %s", error)
            return False

        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    config_manager = ConfigManager()

    logger.info("Configuration Summary:")
    for section, data in config_manager.get_config_summary().items():
        logger.info("  %s: %s", section, data)

    logger.info("Configuration valid: %s", config_manager.validate_config())
