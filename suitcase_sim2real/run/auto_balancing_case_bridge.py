#!/usr/bin/env python3
"""
Auto Balancing Case Sim2Real Bridge
Isaac Lab에서 학습한 RL Policy를 실제 Auto Balancing Case 하드웨어로 실행하는 브릿지
"""

import torch
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pickle
import yaml
from collections import deque

from dynamixel_xl430_interface import DynamixelXL430Interface
from hx711_interface import HX711LoadCellInterface
from policy_interface import PolicyInterface, PolicyConfig
from config_manager import ConfigManager

class AutoBalancingCaseBridge:
    """Auto Balancing Case용 Sim2Real 브릿지"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 설정 관리자 초기화
        self.config_manager = ConfigManager(config_path)
        
        # 설정 유효성 검사
        if not self.config_manager.validate_config():
            raise ValueError("Configuration validation failed")
        
        # 각 인터페이스별 설정 참조
        self.policy_config = self.config_manager.policy
        self.sensor_config = self.config_manager.sensor
        self.actuator_config = self.config_manager.actuator
        self.system_config = self.config_manager.system
        
        # Policy 인터페이스 초기화
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
        
        # Hardware interfaces 초기화
        self._initialize_hardware()
        
        # Observation history 초기화
        self.obs_history = deque(maxlen=self.policy_config.obs_history_length)
        self.action_history = deque(maxlen=self.policy_config.obs_history_length)
        
        # 제어 주기 설정
        self.control_dt = 1.0 / self.policy_config.control_frequency
        
        # 상태 변수들
        self.episode_step = 0
        self.is_running = False
        self.emergency_stop = False
        
        print("Auto Balancing Case Sim2Real Bridge 초기화 완료")
        print("Config Summary:", self.config_manager.get_config_summary())
        print("Policy Info:", self.policy_interface.get_model_info())
        
    def _initialize_hardware(self):
        """하드웨어 인터페이스 초기화"""
        # Dynamixel 듀얼 모터 초기화
        self.motor_interface = DynamixelXL430Interface(
            motor_ids=self.actuator_config.motor_ids,
            device_name=self.actuator_config.device,
            baudrate=self.actuator_config.baudrate,
            control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
        )
        
        # Arduino 기반 Load cell 초기화
        self.load_cell_interface = HX711LoadCellInterface(
            arduino_port=self.sensor_config.port,
            baudrate=self.sensor_config.baudrate
        )
        
        # 캘리브레이션 데이터 로드 시도
        try:
            self.load_cell_interface.load_calibration()
            print("Load cell 캘리브레이션 데이터 로드 완료")
        except:
            print("Warning: Load cell 캘리브레이션 데이터 없음. 캘리브레이션을 먼저 수행하세요.")
    
    def _get_current_observation(self) -> Dict[str, np.ndarray]:
        """현재 하드웨어 상태를 observation으로 변환"""
        # Motor observation
        motor_obs = self.motor_interface.get_observations()
        joint_pos_raw = motor_obs['joint_pos'][0]
        joint_vel_raw = motor_obs['joint_vel'][0]
        
        # 라디안 단위로 변환
        joint_pos_rad = self.motor_interface.get_joint_angle_rad()
        joint_vel_rad_s = self.motor_interface.get_joint_velocity_rad_s()
        
        # Load cell observation
        load_cell_obs = self.load_cell_interface.get_observations()
        wheel_forces = load_cell_obs['wheel_forces']  # [FL, FR, RL, RR]
        handle_force = load_cell_obs['handle_force'][0]
        
        # Previous action (마지막 action이 없으면 0으로 초기화)
        if len(self.action_history) > 0:
            prev_action = self.action_history[-1]
        else:
            prev_action = 0.0
        
        # IsaacLab 환경과 동일한 observation 구조로 변환
        obs = {
            'joint_pos': np.array([joint_pos_rad]),           # 1차원 - 밸런싱 조인트 위치
            'joint_vel': np.array([joint_vel_rad_s]),         # 1차원 - 밸런싱 조인트 속도  
            'prev_action': np.array([prev_action]),           # 1차원 - 이전 액션
            'wheel_contact_forces': wheel_forces,             # 4차원 - 4개 바퀴 접촉력 [FL, FR, RL, RR]
            'handle_external_force': np.array([handle_force]) # 1차원 - 핸들 외부 힘
        }  # 총 8차원 per timestep
        
        return obs
    
    def _normalize_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Observation 정규화 (IsaacLab 환경과 동일)"""
        normalized = {}
        
        # Joint position: -max_angle ~ max_angle -> 정규화 (IsaacLab에서는 relative position 사용)
        normalized['joint_pos'] = obs['joint_pos'] / self.policy_config.max_joint_angle
        
        # Joint velocity: -max_velocity ~ max_velocity -> clipping  
        normalized['joint_vel'] = np.clip(obs['joint_vel'] / self.policy_config.max_joint_velocity, -1.0, 1.0)
        
        # Previous action: 이미 -1~1 범위 (hinge position command)
        normalized['prev_action'] = obs['prev_action']
        
        # Wheel contact forces: 0 ~ max_force -> 0 ~ 1
        normalized['wheel_contact_forces'] = np.clip(obs['wheel_contact_forces'] / self.policy_config.max_wheel_force, 0.0, 1.0)
        
        # Handle external force: 0 ~ max_force -> 0 ~ 1  
        normalized['handle_external_force'] = np.clip(obs['handle_external_force'] / self.policy_config.max_handle_force, 0.0, 1.0)
        
        return normalized
    
    def _check_safety(self, obs: Dict[str, np.ndarray]) -> bool:
        """안전성 체크"""
        joint_angle = obs['joint_pos'][0]
        
        # 각도 제한 체크
        if abs(joint_angle) > self.system_config.emergency_angle_limit:
            print(f"Emergency stop: Joint angle {joint_angle:.3f} rad exceeds limit {self.system_config.emergency_angle_limit:.3f} rad")
            return False
        
        # 에피소드 길이 체크
        if self.episode_step >= self.system_config.max_episode_steps:
            print(f"Episode finished: Max steps {self.system_config.max_episode_steps} reached")
            return False
        
        return True
    
    def run_episode(self):
        """한 에피소드 실행"""
        print("Starting Auto Balancing Case episode...")
        
        # Hardware 시작
        self.motor_interface.start_real_time_reading()
        self.load_cell_interface.start_real_time_reading()
        
        # 초기 위치로 이동 (중앙 위치)
        print("Moving to center position...")
        self.motor_interface.set_angle_command(0.0)
        time.sleep(3.0)  # 초기 위치 도달 대기
        
        # 상태 초기화
        self.episode_step = 0
        self.is_running = True
        self.emergency_stop = False
        self.obs_history.clear()
        self.action_history.clear()
        
        # 초기 observation history 구성
        print("Initializing observation history...")
        for _ in range(self.policy_config.obs_history_length):
            obs = self._get_current_observation()
            normalized_obs = self._normalize_observation(obs)
            self.obs_history.append(normalized_obs)
            self.action_history.append(0.0)
            time.sleep(0.02)  # 20ms 대기
        
        print("Episode started!")
        
        try:
            while self.is_running and not self.emergency_stop:
                step_start_time = time.time()
                
                # 1. 현재 상태 읽기
                raw_obs = self._get_current_observation()
                normalized_obs = self._normalize_observation(raw_obs)
                
                # 2. 안전성 체크
                if not self._check_safety(raw_obs):
                    self.emergency_stop = True
                    break
                
                # 3. Observation history 업데이트
                self.obs_history.append(normalized_obs)
                
                # 4. Policy 실행 (PolicyInterface 사용)
                try:
                    # observation history를 리스트로 변환
                    obs_history_list = list(self.obs_history)
                    action = self.policy_interface.predict(obs_history_list)
                except Exception as e:
                    print(f"Policy prediction error: {e}")
                    action = 0.0  # 안전한 기본값
                
                # 5. Action history 업데이트
                self.action_history.append(action)
                
                # 6. Motor 명령 전송
                # Action은 -1~1 범위이므로 각도로 변환
                target_angle = action * self.policy_config.max_joint_angle
                self.motor_interface.set_angle_command(target_angle)
                
                # 7. 디버깅 정보 출력
                if self.episode_step % 25 == 0:  # 0.5초마다 출력 (50Hz 기준)
                    current_angle = raw_obs['joint_pos'][0]
                    wheel_forces = raw_obs['wheel_contact_forces']
                    handle_force = raw_obs['handle_external_force'][0]
                    
                    print(f"Step {self.episode_step:4d}: "
                          f"Angle={current_angle:+.3f}rad({current_angle*180/np.pi:+.1f}°) "
                          f"Target={target_angle:+.3f}rad({target_angle*180/np.pi:+.1f}°) "
                          f"Action={action:+.3f} "
                          f"Wheels=[{wheel_forces[0]:.1f},{wheel_forces[1]:.1f},{wheel_forces[2]:.1f},{wheel_forces[3]:.1f}]N "
                          f"Handle={handle_force:.1f}N")
                
                # 8. 제어 주기 유지
                elapsed = time.time() - step_start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:  # 10ms 이상 지연 시 경고
                    print(f"Warning: Control loop delay: {-sleep_time:.3f}s")
                
                self.episode_step += 1
        
        except KeyboardInterrupt:
            print("\nEpisode interrupted by user")
            self.emergency_stop = True
        
        except Exception as e:
            print(f"Episode error: {e}")
            self.emergency_stop = True
        
        finally:
            print("Episode finished. Shutting down...")
            
            # 안전한 위치로 이동
            self.motor_interface.set_angle_command(0.0)
            time.sleep(2.0)
            
            # Hardware 정지
            self.motor_interface.stop_real_time_reading()
            self.load_cell_interface.stop_real_time_reading()
            
            print(f"Episode completed: {self.episode_step} steps")
    
    def run_continuous(self):
        """연속 실행 (에피소드 개념 없이)"""
        print("Starting continuous Auto Balancing Case control...")
        
        # Hardware 시작
        self.motor_interface.start_real_time_reading()
        self.load_cell_interface.start_real_time_reading()
        
        # 초기 위치로 이동
        print("Moving to center position...")
        self.motor_interface.set_angle_command(0.0)
        time.sleep(3.0)
        
        # 상태 초기화
        step_count = 0
        self.obs_history.clear()
        self.action_history.clear()
        
        # 초기 observation history 구성
        print("Initializing observation history...")
        for _ in range(self.policy_config.obs_history_length):
            obs = self._get_current_observation()
            normalized_obs = self._normalize_observation(obs)
            self.obs_history.append(normalized_obs)
            self.action_history.append(0.0)
            time.sleep(0.02)
        
        print("Continuous control started!")
        
        try:
            while True:
                step_start_time = time.time()
                
                # 현재 상태 읽기
                raw_obs = self._get_current_observation()
                normalized_obs = self._normalize_observation(raw_obs)
                
                # 안전성 체크 (각도만)
                joint_angle = raw_obs['joint_pos'][0]
                if abs(joint_angle) > self.system_config.emergency_angle_limit:
                    print(f"Emergency stop: Joint angle {joint_angle:.3f} rad exceeds limit")
                    break
                
                # Observation history 업데이트
                self.obs_history.append(normalized_obs)
                
                # Policy 실행 (PolicyInterface 사용)
                try:
                    obs_history_list = list(self.obs_history)
                    action = self.policy_interface.predict(obs_history_list)
                except Exception as e:
                    print(f"Policy prediction error: {e}")
                    action = 0.0
                
                # Action history 업데이트
                self.action_history.append(action)
                
                # Motor 명령 전송
                target_angle = action * self.policy_config.max_joint_angle
                self.motor_interface.set_angle_command(target_angle)
                
                # 모니터링
                if step_count % 50 == 0:  # 1초마다 출력
                    current_angle = raw_obs['joint_pos'][0]
                    print(f"Step {step_count}: "
                          f"Angle={current_angle:+.3f}rad "
                          f"Target={target_angle:+.3f}rad "
                          f"Action={action:+.3f}")
                
                # 제어 주기 유지
                elapsed = time.time() - step_start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                step_count += 1
        
        except KeyboardInterrupt:
            print("\nContinuous control stopped by user")
        
        finally:
            # 안전한 위치로 이동
            self.motor_interface.set_angle_command(0.0)
            time.sleep(2.0)
            
            # Hardware 정지
            self.motor_interface.stop_real_time_reading()
            self.load_cell_interface.stop_real_time_reading()
    
    def calibrate_load_cells(self):
        """Load cell 캘리브레이션 수행"""
        print("Starting load cell calibration...")
        self.load_cell_interface.calibrate_all_load_cells(200.0)  # 1kg 추 사용
        self.load_cell_interface.save_calibration()
    
    def shutdown(self):
        """정리 및 종료"""
        self.is_running = False
        self.motor_interface.shutdown()
        self.load_cell_interface.shutdown()
        print("Auto Balancing Case Bridge shutdown complete")


# 메인 실행 예제
if __name__ == "__main__":
    # ConfigManager를 사용한 Bridge 초기화
    # 기본 config 파일 사용: config/interface_config.yml
    bridge = AutoBalancingCaseBridge()
    
    try:
        # Load cell 캘리브레이션 (처음 한 번만)
        # bridge.calibrate_load_cells()
        
        # 한 에피소드 실행
        bridge.run_episode()
        
        # 또는 연속 실행
        # bridge.run_continuous()
    
    finally:
        bridge.shutdown()
