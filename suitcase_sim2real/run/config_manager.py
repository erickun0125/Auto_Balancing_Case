#!/usr/bin/env python3
"""
Configuration Manager for Auto Balancing Case
interface_config.yml 파일을 로드하고 각 인터페이스별 설정을 관리하는 모듈
"""

import yaml
import os
import platform
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyInterfaceConfig:
    """Policy Interface 설정"""
    model_path: str
    device: str = "cpu"
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    obs_history_length: int = 4
    obs_dim_per_timestep: int = 8
    num_actions: int = 1
    control_frequency: float = 50.0
    
    # Normalization parameters
    max_joint_angle: float = 0.5
    max_joint_velocity: float = 6.0
    max_wheel_force: float = 50.0
    max_handle_force: float = 20.0


@dataclass
class SensorInterfaceConfig:
    """Sensor Interface 설정"""
    # Load cell connection
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    timeout: float = 2.0
    
    # Calibration
    auto_load_on_startup: bool = True
    calibration_file: str = "load_cell_calibration.npz"
    reference_weight: float = 200.0
    
    # Sensors
    wheel_sensor_count: int = 4
    wheel_sensor_names: list = field(default_factory=lambda: ["FL", "FR", "RL", "RR"])
    wheel_channels: list = field(default_factory=lambda: [0, 1, 2, 3])
    handle_sensor_count: int = 1
    handle_sensor_names: list = field(default_factory=lambda: ["handle"])
    handle_channels: list = field(default_factory=lambda: [4])
    
    # Processing
    sampling_rate: float = 100.0
    filter_enabled: bool = True
    filter_type: str = "moving_average"
    filter_window: int = 5
    noise_threshold: float = 0.1


@dataclass
class ActuatorInterfaceConfig:
    """Actuator Interface 설정"""
    # Motor connection
    device: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    protocol_version: float = 2.0
    timeout: float = 1.0
    
    # Motor configuration
    motor_ids: list = field(default_factory=lambda: [1, 2])
    control_mode: str = "position"
    profile_velocity: int = 50
    profile_acceleration: int = 20
    
    # Safety limits
    position_limit_min: float = -0.5
    position_limit_max: float = 0.5
    velocity_limit: float = 6.0
    current_limit: int = 500
    temperature_limit: int = 80
    
    # Control parameters
    position_p: int = 800
    position_i: int = 0
    position_d: int = 0
    
    # Monitoring
    monitoring_enabled: bool = True
    monitoring_frequency: float = 100.0
    log_data: bool = False


@dataclass
class SystemConfig:
    """System 전역 설정"""
    # Safety
    emergency_stop_enabled: bool = True
    emergency_angle_limit: float = 0.4
    emergency_force_limit: float = 100.0
    watchdog_enabled: bool = True
    watchdog_timeout: float = 1.0
    
    # Session
    max_episode_steps: int = 400
    episode_timeout: float = 8.0
    auto_reset: bool = True
    
    # Logging
    logging_enabled: bool = True
    log_level: str = "INFO"
    log_file: str = "auto_balancing_case.log"
    console_output: bool = True
    data_logging_enabled: bool = False
    data_logging_frequency: float = 50.0
    data_output_dir: str = "logs/data"
    data_format: str = "csv"
    
    # Performance
    monitor_timing: bool = True
    target_control_frequency: float = 50.0
    timing_warning_threshold: float = 0.02
    timing_error_threshold: float = 0.05


class ConfigManager:
    """설정 파일 관리자"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 기본 config 파일 경로
            current_dir = Path(__file__).parent
            config_path = str(current_dir / "config" / "interface_config.yml")
        
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self._apply_platform_overrides()
        
        # 각 인터페이스별 설정 객체 생성
        self.policy = self._create_policy_config()
        self.sensor = self._create_sensor_config()
        self.actuator = self._create_actuator_config()
        self.system = self._create_system_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _apply_platform_overrides(self):
        """플랫폼별 설정 오버라이드 적용"""
        current_platform = platform.system().lower()
        
        if 'platform_overrides' not in self.raw_config:
            return
        
        platform_config = self.raw_config['platform_overrides'].get(current_platform, {})
        
        for key_path, value in platform_config.items():
            self._set_nested_value(self.raw_config, key_path, value)
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any):
        """중첩된 딕셔너리에 값 설정 (예: "sensor.load_cell.connection.port")"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _create_policy_config(self) -> PolicyInterfaceConfig:
        """Policy Interface 설정 객체 생성"""
        policy_cfg = self.raw_config.get('policy', {})
        norm_cfg = policy_cfg.get('normalization', {})
        
        return PolicyInterfaceConfig(
            model_path=policy_cfg.get('model_path', ''),
            device=policy_cfg.get('device', 'cpu'),
            actor_hidden_dims=policy_cfg.get('actor_hidden_dims', [256, 128, 64]),
            critic_hidden_dims=policy_cfg.get('critic_hidden_dims', [256, 128, 64]),
            activation=policy_cfg.get('activation', 'elu'),
            init_noise_std=policy_cfg.get('init_noise_std', 1.0),
            noise_std_type=policy_cfg.get('noise_std_type', 'scalar'),
            obs_history_length=policy_cfg.get('obs_history_length', 4),
            obs_dim_per_timestep=policy_cfg.get('obs_dim_per_timestep', 8),
            num_actions=policy_cfg.get('num_actions', 1),
            control_frequency=policy_cfg.get('control_frequency', 50.0),
            max_joint_angle=norm_cfg.get('max_joint_angle', 0.5),
            max_joint_velocity=norm_cfg.get('max_joint_velocity', 6.0),
            max_wheel_force=norm_cfg.get('max_wheel_force', 50.0),
            max_handle_force=norm_cfg.get('max_handle_force', 20.0)
        )
    
    def _create_sensor_config(self) -> SensorInterfaceConfig:
        """Sensor Interface 설정 객체 생성"""
        sensor_cfg = self.raw_config.get('sensor', {})
        load_cell_cfg = sensor_cfg.get('load_cell', {})
        conn_cfg = load_cell_cfg.get('connection', {})
        calib_cfg = load_cell_cfg.get('calibration', {})
        sensors_cfg = load_cell_cfg.get('sensors', {})
        wheel_cfg = sensors_cfg.get('wheel_sensors', {})
        handle_cfg = sensors_cfg.get('handle_sensor', {})
        proc_cfg = load_cell_cfg.get('processing', {})
        
        return SensorInterfaceConfig(
            port=conn_cfg.get('port', '/dev/ttyACM0'),
            baudrate=conn_cfg.get('baudrate', 115200),
            timeout=conn_cfg.get('timeout', 2.0),
            auto_load_on_startup=calib_cfg.get('auto_load_on_startup', True),
            calibration_file=calib_cfg.get('calibration_file', 'load_cell_calibration.npz'),
            reference_weight=calib_cfg.get('reference_weight', 200.0),
            wheel_sensor_count=wheel_cfg.get('count', 4),
            wheel_sensor_names=wheel_cfg.get('names', ['FL', 'FR', 'RL', 'RR']),
            wheel_channels=wheel_cfg.get('channels', [0, 1, 2, 3]),
            handle_sensor_count=handle_cfg.get('count', 1),
            handle_sensor_names=handle_cfg.get('names', ['handle']),
            handle_channels=handle_cfg.get('channels', [4]),
            sampling_rate=proc_cfg.get('sampling_rate', 100.0),
            filter_enabled=proc_cfg.get('filter_enabled', True),
            filter_type=proc_cfg.get('filter_type', 'moving_average'),
            filter_window=proc_cfg.get('filter_window', 5),
            noise_threshold=proc_cfg.get('noise_threshold', 0.1)
        )
    
    def _create_actuator_config(self) -> ActuatorInterfaceConfig:
        """Actuator Interface 설정 객체 생성"""
        actuator_cfg = self.raw_config.get('actuator', {})
        motor_cfg = actuator_cfg.get('motor', {})
        conn_cfg = motor_cfg.get('connection', {})
        motors_cfg = motor_cfg.get('motors', {})
        balancing_cfg = motors_cfg.get('balancing_joint', {})
        limits_cfg = motor_cfg.get('limits', {})
        control_cfg = motor_cfg.get('control', {})
        pid_cfg = control_cfg.get('pid_gains', {})
        monitor_cfg = control_cfg.get('monitoring', {})
        
        return ActuatorInterfaceConfig(
            device=conn_cfg.get('device', '/dev/ttyUSB0'),
            baudrate=conn_cfg.get('baudrate', 57600),
            protocol_version=conn_cfg.get('protocol_version', 2.0),
            timeout=conn_cfg.get('timeout', 1.0),
            motor_ids=balancing_cfg.get('ids', [1, 2]),
            control_mode=balancing_cfg.get('control_mode', 'position'),
            profile_velocity=balancing_cfg.get('profile_velocity', 50),
            profile_acceleration=balancing_cfg.get('profile_acceleration', 20),
            position_limit_min=limits_cfg.get('position_limit', {}).get('min', -0.5),
            position_limit_max=limits_cfg.get('position_limit', {}).get('max', 0.5),
            velocity_limit=limits_cfg.get('velocity_limit', 6.0),
            current_limit=limits_cfg.get('current_limit', 500),
            temperature_limit=limits_cfg.get('temperature_limit', 80),
            position_p=pid_cfg.get('position_p', 800),
            position_i=pid_cfg.get('position_i', 0),
            position_d=pid_cfg.get('position_d', 0),
            monitoring_enabled=monitor_cfg.get('enabled', True),
            monitoring_frequency=monitor_cfg.get('frequency', 100.0),
            log_data=monitor_cfg.get('log_data', False)
        )
    
    def _create_system_config(self) -> SystemConfig:
        """System 설정 객체 생성"""
        system_cfg = self.raw_config.get('system', {})
        safety_cfg = system_cfg.get('safety', {})
        emergency_cfg = safety_cfg.get('emergency_stop', {})
        watchdog_cfg = safety_cfg.get('watchdog', {})
        session_cfg = system_cfg.get('session', {})
        logging_cfg = system_cfg.get('logging', {})
        data_logging_cfg = logging_cfg.get('data_logging', {})
        perf_cfg = system_cfg.get('performance', {})
        
        return SystemConfig(
            emergency_stop_enabled=emergency_cfg.get('enabled', True),
            emergency_angle_limit=emergency_cfg.get('angle_limit', 0.4),
            emergency_force_limit=emergency_cfg.get('force_limit', 100.0),
            watchdog_enabled=watchdog_cfg.get('enabled', True),
            watchdog_timeout=watchdog_cfg.get('timeout', 1.0),
            max_episode_steps=session_cfg.get('max_episode_steps', 400),
            episode_timeout=session_cfg.get('episode_timeout', 8.0),
            auto_reset=session_cfg.get('auto_reset', True),
            logging_enabled=logging_cfg.get('enabled', True),
            log_level=logging_cfg.get('level', 'INFO'),
            log_file=logging_cfg.get('log_file', 'auto_balancing_case.log'),
            console_output=logging_cfg.get('console_output', True),
            data_logging_enabled=data_logging_cfg.get('enabled', False),
            data_logging_frequency=data_logging_cfg.get('frequency', 50.0),
            data_output_dir=data_logging_cfg.get('output_dir', 'logs/data'),
            data_format=data_logging_cfg.get('format', 'csv'),
            monitor_timing=perf_cfg.get('monitor_timing', True),
            target_control_frequency=perf_cfg.get('target_control_frequency', 50.0),
            timing_warning_threshold=perf_cfg.get('timing_warning_threshold', 0.02),
            timing_error_threshold=perf_cfg.get('timing_error_threshold', 0.05)
        )
    
    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보 반환"""
        return {
            "config_file": str(self.config_path),
            "platform": platform.system(),
            "policy": {
                "model_path": self.policy.model_path,
                "device": self.policy.device,
                "control_frequency": self.policy.control_frequency
            },
            "sensor": {
                "port": self.sensor.port,
                "baudrate": self.sensor.baudrate
            },
            "actuator": {
                "device": self.actuator.device,
                "motor_ids": self.actuator.motor_ids
            },
            "system": {
                "max_episode_steps": self.system.max_episode_steps,
                "emergency_angle_limit": self.system.emergency_angle_limit
            }
        }
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        errors = []
        
        # Policy 검사
        if not os.path.exists(self.policy.model_path):
            errors.append(f"Policy model file not found: {self.policy.model_path}")
        
        if self.policy.control_frequency <= 0:
            errors.append(f"Invalid control frequency: {self.policy.control_frequency}")
        
        # 센서 포트 존재 확인 (Linux/Windows에 따라)
        # if not os.path.exists(self.sensor.port):
        #     errors.append(f"Sensor port not found: {self.sensor.port}")
        
        # 액츄에이터 포트 존재 확인
        # if not os.path.exists(self.actuator.device):
        #     errors.append(f"Actuator device not found: {self.actuator.device}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# 사용 예제
if __name__ == "__main__":
    # 설정 관리자 생성
    config_manager = ConfigManager()
    
    # 설정 요약 출력
    print("Configuration Summary:")
    summary = config_manager.get_config_summary()
    for section, data in summary.items():
        print(f"  {section}: {data}")
    
    # 설정 유효성 검사
    print(f"\nConfiguration valid: {config_manager.validate_config()}")
    
    # 개별 설정 접근 예제
    print(f"\nPolicy model path: {config_manager.policy.model_path}")
    print(f"Sensor port: {config_manager.sensor.port}")
    print(f"Motor IDs: {config_manager.actuator.motor_ids}")
    print(f"Emergency angle limit: {config_manager.system.emergency_angle_limit}")
