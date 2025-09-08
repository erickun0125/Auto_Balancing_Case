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
        self.initial_joint_pos = 0.0  # Isaac Lab의 relative position 계산을 위한 초기 위치
        
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
        """현재 하드웨어 상태를 observation으로 변환 (Isaac Lab 환경과 동일)"""
        # Motor state (새로운 통합 API 사용)
        motor_state = self.motor_interface.get_state()
        joint_pos_rad = motor_state['position']     # 이미 라디안 단위
        joint_vel_rad_s = motor_state['velocity']   # 이미 라디안/초 단위
        
        # Load cell observation
        load_cell_obs = self.load_cell_interface.get_observations()
        wheel_forces = load_cell_obs['wheel_forces']  # [FL, FR, RL, RR]
        handle_force = load_cell_obs['handle_force'][0]
        
        # Previous action (마지막 action이 없으면 0으로 초기화)
        if len(self.action_history) > 0:
            prev_action = self.action_history[-1]
        else:
            prev_action = 0.0
        
        # Isaac Lab 환경과 동일한 observation 구조로 변환
        # 중요: Isaac Lab에서는 joint_pos_rel을 사용하므로 초기 위치로부터의 상대 위치를 계산
        relative_joint_pos = joint_pos_rad - self.initial_joint_pos
        
        obs = {
            'joint_pos': np.array([relative_joint_pos]),       # relative position (현재 - 초기위치)
            'joint_vel': np.array([joint_vel_rad_s]),          # joint velocity 
            'prev_action': np.array([prev_action]),            # 이전 액션 (이미 -0.5~0.5 범위)
            'wheel_contact_forces': wheel_forces,              # 4차원 - 4개 바퀴 접촉력 [FL, FR, RL, RR]
            'handle_external_force': np.array([handle_force])  # 1차원 - 핸들 외부 힘
        }  # 총 8차원 per timestep
        
        return obs
    
    # 더 이상 사용되지 않음 - Isaac Lab에서는 observation을 정규화하지 않음
    # def _normalize_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    #     """Observation 정규화 (사용되지 않음 - Isaac Lab은 raw 값 사용)"""
    #     pass
    
    def _check_episode_length(self) -> bool:
        """에피소드 길이 체크"""
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
        self.motor_interface.set_command(0.0)
        time.sleep(3.0)  # 초기 위치 도달 대기
        
        # 초기 위치 저장 (Isaac Lab relative position 계산용)
        motor_state = self.motor_interface.get_state()
        self.initial_joint_pos = motor_state['position']
        print(f"Initial joint position set to: {self.initial_joint_pos:.3f} rad")
        
        # 상태 초기화
        self.episode_step = 0
        self.is_running = True
        self.emergency_stop = False
        self.obs_history.clear()
        self.action_history.clear()
        
        # 초기 observation history 구성 (raw 값 사용, 정규화하지 않음)
        print("Initializing observation history...")
        for _ in range(self.policy_config.obs_history_length):
            obs = self._get_current_observation()
            self.obs_history.append(obs)  # raw 값 그대로 사용
            self.action_history.append(0.0)
            time.sleep(0.02)  # 20ms 대기
        
        print("Episode started!")
        
        try:
            while self.is_running and not self.emergency_stop:
                step_start_time = time.time()
                
                # 1. 현재 상태 읽기 (raw 값 사용, 정규화하지 않음)
                raw_obs = self._get_current_observation()
                
                # 2. 안전성 체크 (절대 위치 기준)
                motor_state = self.motor_interface.get_state()
                current_joint_pos_abs = motor_state['position']
                if abs(current_joint_pos_abs) > self.system_config.emergency_angle_limit:
                    print(f"Emergency stop: Joint angle {current_joint_pos_abs:.3f} rad exceeds limit {self.system_config.emergency_angle_limit:.3f} rad")
                    self.emergency_stop = True
                    break
                
                # 에피소드 길이 체크
                if not self._check_episode_length():
                    self.emergency_stop = True
                    break
                
                # 3. Observation history 업데이트 (정규화하지 않고 raw 값 사용)
                self.obs_history.append(raw_obs)
                
                # 4. Policy 실행 (PolicyInterface 사용)
                try:
                    # observation history를 리스트로 변환 (정규화하지 않고 raw 값 사용)
                    obs_history_list = list(self.obs_history)
                    raw_action = self.policy_interface.predict(obs_history_list)
                    
                    # Isaac Lab과 동일한 action clipping 적용
                    action = np.clip(raw_action, -0.5, 0.5)  # Isaac Lab의 clip 범위와 동일
                    
                except Exception as e:
                    print(f"Policy prediction error: {e}")
                    action = 0.0  # 안전한 기본값
                
                # 5. Action history 업데이트
                self.action_history.append(action)
                
                # 6. Motor 명령 전송
                # Isaac Lab에서 action은 이미 -0.5~0.5 rad 범위로 clipping되어 나옴
                # scale=1.0이고 clip={BALANCE_JOINT_NAME: (-0.5, 0.5)}이므로 그대로 사용
                target_angle = action  # action은 이미 라디안 단위 (-0.5~0.5 rad)
                self.motor_interface.set_command(target_angle)
                
                # 7. 디버깅 정보 출력
                if self.episode_step % 25 == 0:  # 0.5초마다 출력 (50Hz 기준)
                    current_angle_abs = current_joint_pos_abs  # 절대 위치
                    current_angle_rel = raw_obs['joint_pos'][0]  # 상대 위치 (정책에 전달되는 값)
                    wheel_forces = raw_obs['wheel_contact_forces']
                    handle_force = raw_obs['handle_external_force'][0]
                    
                    print(f"Step {self.episode_step:4d}: "
                          f"Abs={current_angle_abs:+.3f}rad({current_angle_abs*180/np.pi:+.1f}°) "
                          f"Rel={current_angle_rel:+.3f}rad({current_angle_rel*180/np.pi:+.1f}°) "
                          f"RawAction={raw_action:+.3f} "
                          f"ClippedAction={action:+.3f} "
                          f"Target={target_angle:+.3f}rad({target_angle*180/np.pi:+.1f}°) "
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
            self.motor_interface.set_command(0.0)
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
        self.motor_interface.set_command(0.0)
        time.sleep(3.0)
        
        # 초기 위치 저장 (Isaac Lab relative position 계산용)
        motor_state = self.motor_interface.get_state()
        self.initial_joint_pos = motor_state['position']
        print(f"Initial joint position set to: {self.initial_joint_pos:.3f} rad")
        
        # 상태 초기화
        step_count = 0
        self.obs_history.clear()
        self.action_history.clear()
        
        # 초기 observation history 구성 (raw 값 사용, 정규화하지 않음)
        print("Initializing observation history...")
        for _ in range(self.policy_config.obs_history_length):
            obs = self._get_current_observation()
            self.obs_history.append(obs)  # raw 값 그대로 사용
            self.action_history.append(0.0)
            time.sleep(0.02)
        
        print("Continuous control started!")
        
        try:
            while True:
                step_start_time = time.time()
                
                # 현재 상태 읽기 (raw 값 사용, 정규화하지 않음)
                raw_obs = self._get_current_observation()
                
                # 안전성 체크 (절대 위치 기준)
                motor_state = self.motor_interface.get_state()
                current_joint_pos_abs = motor_state['position']
                if abs(current_joint_pos_abs) > self.system_config.emergency_angle_limit:
                    print(f"Emergency stop: Joint angle {current_joint_pos_abs:.3f} rad exceeds limit")
                    break
                
                # Observation history 업데이트 (raw 값 사용)
                self.obs_history.append(raw_obs)
                
                # Policy 실행 (PolicyInterface 사용)
                try:
                    obs_history_list = list(self.obs_history)
                    raw_action = self.policy_interface.predict(obs_history_list)
                    
                    # Isaac Lab과 동일한 action clipping 적용
                    action = np.clip(raw_action, -0.5, 0.5)
                    
                except Exception as e:
                    print(f"Policy prediction error: {e}")
                    action = 0.0
                
                # Action history 업데이트
                self.action_history.append(action)
                
                # Motor 명령 전송
                # Isaac Lab에서 action은 이미 -0.5~0.5 rad 범위로 clipping되어 나옴
                target_angle = action  # action은 이미 라디안 단위 (-0.5~0.5 rad)
                self.motor_interface.set_command(target_angle)
                
                # 모니터링
                if step_count % 50 == 0:  # 1초마다 출력
                    current_angle_abs = current_joint_pos_abs  # 절대 위치
                    current_angle_rel = raw_obs['joint_pos'][0]  # 상대 위치
                    print(f"Step {step_count}: "
                          f"Abs={current_angle_abs:+.3f}rad "
                          f"Rel={current_angle_rel:+.3f}rad "
                          f"RawAction={raw_action:+.3f} "
                          f"ClippedAction={action:+.3f} "
                          f"Target={target_angle:+.3f}rad")
                
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
            self.motor_interface.set_command(0.0)
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
