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

@dataclass
class AutoBalancingCaseConfig:
    """Auto Balancing Case 설정"""
    # Policy 설정
    model_path: str
    control_frequency: float = 50.0  # Hz (IsaacLab decimation=4, dt=0.005 -> 50Hz)
    device: str = "cpu"
    
    # Hardware 설정 - Dual Motors
    motor_ids: List[int] = None  # [1, 2] - 듀얼 모터 ID
    motor_device: str = '/dev/ttyUSB0'
    motor_baudrate: int = 57600
    
    # Load cell 설정 - Arduino 기반
    arduino_port: str = '/dev/ttyACM0'
    arduino_baudrate: int = 115200
    
    # Observation history 설정 (IsaacLab에서 사용하는 history length)
    obs_history_length: int = 4
    
    # 정규화 파라미터
    max_wheel_force: float = 50.0  # N
    max_handle_force: float = 20.0  # N
    max_joint_angle: float = 0.5   # rad
    max_joint_velocity: float = 6.0  # rad/s
    
    # Safety 설정
    max_episode_steps: int = 400  # 50Hz * 8초 (IsaacLab episode_length_s=8.0과 일치)
    emergency_angle_limit: float = 0.4  # rad (비상 정지 각도)

class AutoBalancingCaseBridge:
    """Auto Balancing Case용 Sim2Real 브릿지"""
    
    def __init__(self, config: AutoBalancingCaseConfig):
        self.config = config
        
        # Policy 로드
        self.policy = self._load_policy()
        
        # Hardware interfaces 초기화
        self._initialize_hardware()
        
        # Observation history 초기화
        self.obs_history = deque(maxlen=config.obs_history_length)
        self.action_history = deque(maxlen=config.obs_history_length)
        
        # 제어 주기 설정
        self.control_dt = 1.0 / config.control_frequency
        
        # 상태 변수들
        self.episode_step = 0
        self.is_running = False
        self.emergency_stop = False
        
        print("Auto Balancing Case Sim2Real Bridge 초기화 완료")
        
    def _initialize_hardware(self):
        """하드웨어 인터페이스 초기화"""
        # Dynamixel 듀얼 모터 초기화
        if self.config.motor_ids is None:
            self.config.motor_ids = [1, 2]  # 기본값: 모터 ID 1, 2
            
        self.motor_interface = DynamixelXL430Interface(
            motor_ids=self.config.motor_ids,
            device_name=self.config.motor_device,
            baudrate=self.config.motor_baudrate,
            control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
        )
        
        # Arduino 기반 Load cell 초기화
        self.load_cell_interface = HX711LoadCellInterface(
            arduino_port=self.config.arduino_port,
            baudrate=self.config.arduino_baudrate
        )
        
        # 캘리브레이션 데이터 로드 시도
        try:
            self.load_cell_interface.load_calibration()
            print("Load cell 캘리브레이션 데이터 로드 완료")
        except:
            print("Warning: Load cell 캘리브레이션 데이터 없음. 캘리브레이션을 먼저 수행하세요.")
    
    def _load_policy(self) -> torch.nn.Module:
        """Isaac Lab RSL-RL policy 로드"""
        print(f"Loading policy from {self.config.model_path}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Policy file not found: {self.config.model_path}")
        
        try:
            # RSL-RL checkpoint 로드
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
            
            # RSL-RL 표준 구조에서 policy 추출
            if 'model_state_dict' in checkpoint:
                policy_state_dict = checkpoint['model_state_dict']
            elif 'ac_weights' in checkpoint:
                policy_state_dict = checkpoint['ac_weights']
            elif 'policy_state_dict' in checkpoint:
                policy_state_dict = checkpoint['policy_state_dict']
            else:
                # 직접 state_dict인 경우
                policy_state_dict = checkpoint
            
            # Policy 네트워크 구조 생성 (IsaacLab RSL-RL 기본 구조)
            policy = self._create_policy_network()
            
            # Actor 부분만 로드 (critic은 필요 없음)
            actor_state_dict = {}
            for key, value in policy_state_dict.items():
                if 'actor' in key:
                    # 'actor.' 접두사 제거
                    new_key = key.replace('actor.', '')
                    actor_state_dict[new_key] = value
            
            policy.load_state_dict(actor_state_dict, strict=False)
            policy.eval()
            
            print("Policy loaded successfully")
            return policy
            
        except Exception as e:
            print(f"Policy loading error: {e}")
            raise
    
    def _create_policy_network(self) -> torch.nn.Module:
        """Policy 네트워크 구조 생성 (IsaacLab RSL-RL 기본 MLP 구조)"""
        import torch.nn as nn
        
        # Observation dimensions 계산
        # joint_pos (1) + joint_vel (1) + prev_action (1) + wheel_forces (4) + handle_force (1) = 8
        # history_length=4이므로 총 8 * 4 = 32
        obs_dim = 8 * self.config.obs_history_length
        action_dim = 1  # 단일 밸런싱 조인트
        
        # IsaacLab RSL-RL PPO 기본 구조 (rsl_rl_ppo_cfg.py 참조)
        hidden_dims = [256, 128, 64]
        
        class MLPActor(nn.Module):
            def __init__(self, obs_dim, action_dim, hidden_dims):
                super().__init__()
                layers = []
                in_dim = obs_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.ELU()  # RSL-RL에서 사용하는 activation
                    ])
                    in_dim = hidden_dim
                
                layers.append(nn.Linear(in_dim, action_dim))
                layers.append(nn.Tanh())  # action 범위 -1~1
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, obs):
                return self.network(obs)
        
        policy = MLPActor(obs_dim, action_dim, hidden_dims)
        return policy
    
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
        
        # Observation 구성 (IsaacLab 환경과 동일한 구조)
        obs = {
            'joint_pos': np.array([joint_pos_rad]),
            'joint_vel': np.array([joint_vel_rad_s]),
            'prev_action': np.array([prev_action]),
            'wheel_forces': wheel_forces,
            'handle_force': np.array([handle_force])
        }
        
        return obs
    
    def _normalize_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Observation 정규화 (IsaacLab 환경과 동일)"""
        normalized = {}
        
        # Joint position: -max_angle ~ max_angle -> -1 ~ 1
        normalized['joint_pos'] = obs['joint_pos'] / self.config.max_joint_angle
        
        # Joint velocity: -max_velocity ~ max_velocity -> -1 ~ 1  
        normalized['joint_vel'] = np.clip(obs['joint_vel'] / self.config.max_joint_velocity, -1.0, 1.0)
        
        # Previous action: 이미 -1~1 범위
        normalized['prev_action'] = obs['prev_action']
        
        # Wheel forces: 0 ~ max_force -> 0 ~ 1
        normalized['wheel_forces'] = np.clip(obs['wheel_forces'] / self.config.max_wheel_force, 0.0, 1.0)
        
        # Handle force: 0 ~ max_force -> 0 ~ 1
        normalized['handle_force'] = np.clip(obs['handle_force'] / self.config.max_handle_force, 0.0, 1.0)
        
        return normalized
    
    def _prepare_policy_input(self) -> torch.Tensor:
        """Policy 입력을 위한 observation history 준비"""
        if len(self.obs_history) == 0:
            return None
        
        # History를 concatenate (가장 오래된 것부터 최신 순)
        obs_vectors = []
        for obs in self.obs_history:
            # 각 observation을 벡터로 변환
            obs_vector = np.concatenate([
                obs['joint_pos'],      # 1
                obs['joint_vel'],      # 1  
                obs['prev_action'],    # 1
                obs['wheel_forces'],   # 4
                obs['handle_force']    # 1
            ])  # 총 8차원
            obs_vectors.append(obs_vector)
        
        # History가 부족한 경우 0으로 패딩
        while len(obs_vectors) < self.config.obs_history_length:
            obs_vectors.insert(0, np.zeros(8))
        
        # Concatenate all history
        policy_input = np.concatenate(obs_vectors)  # 8 * 4 = 32차원
        
        # Convert to torch tensor
        policy_input = torch.from_numpy(policy_input).float().unsqueeze(0)  # batch dimension 추가
        
        return policy_input.to(self.config.device)
    
    def _check_safety(self, obs: Dict[str, np.ndarray]) -> bool:
        """안전성 체크"""
        joint_angle = obs['joint_pos'][0]
        
        # 각도 제한 체크
        if abs(joint_angle) > self.config.emergency_angle_limit:
            print(f"Emergency stop: Joint angle {joint_angle:.3f} rad exceeds limit {self.config.emergency_angle_limit:.3f} rad")
            return False
        
        # 에피소드 길이 체크
        if self.episode_step >= self.config.max_episode_steps:
            print(f"Episode finished: Max steps {self.config.max_episode_steps} reached")
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
        for _ in range(self.config.obs_history_length):
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
                
                # 4. Policy 입력 준비
                policy_input = self._prepare_policy_input()
                
                # 5. Policy 실행
                with torch.no_grad():
                    action_tensor = self.policy(policy_input)
                    action = action_tensor.squeeze().cpu().numpy()
                    
                    # Scalar action인 경우 처리
                    if action.ndim == 0:
                        action = float(action)
                    else:
                        action = action[0]
                
                # 6. Action history 업데이트
                self.action_history.append(action)
                
                # 7. Motor 명령 전송
                # Action은 -1~1 범위이므로 각도로 변환
                target_angle = action * self.config.max_joint_angle
                self.motor_interface.set_angle_command(target_angle)
                
                # 8. 디버깅 정보 출력
                if self.episode_step % 25 == 0:  # 0.5초마다 출력 (50Hz 기준)
                    current_angle = raw_obs['joint_pos'][0]
                    wheel_forces = raw_obs['wheel_forces']
                    handle_force = raw_obs['handle_force'][0]
                    
                    print(f"Step {self.episode_step:4d}: "
                          f"Angle={current_angle:+.3f}rad({current_angle*180/np.pi:+.1f}°) "
                          f"Target={target_angle:+.3f}rad({target_angle*180/np.pi:+.1f}°) "
                          f"Action={action:+.3f} "
                          f"Wheels=[{wheel_forces[0]:.1f},{wheel_forces[1]:.1f},{wheel_forces[2]:.1f},{wheel_forces[3]:.1f}]N "
                          f"Handle={handle_force:.1f}N")
                
                # 9. 제어 주기 유지
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
        for _ in range(self.config.obs_history_length):
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
                if abs(joint_angle) > self.config.emergency_angle_limit:
                    print(f"Emergency stop: Joint angle {joint_angle:.3f} rad exceeds limit")
                    break
                
                # Observation history 업데이트
                self.obs_history.append(normalized_obs)
                
                # Policy 실행
                policy_input = self._prepare_policy_input()
                with torch.no_grad():
                    action_tensor = self.policy(policy_input)
                    action = action_tensor.squeeze().cpu().numpy()
                    
                    if action.ndim == 0:
                        action = float(action)
                    else:
                        action = action[0]
                
                # Action history 업데이트
                self.action_history.append(action)
                
                # Motor 명령 전송
                target_angle = action * self.config.max_joint_angle
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
        self.load_cell_interface.calibrate_all_load_cells(1000.0)  # 1kg 추 사용
        self.load_cell_interface.save_calibration()
    
    def shutdown(self):
        """정리 및 종료"""
        self.is_running = False
        self.motor_interface.shutdown()
        self.load_cell_interface.shutdown()
        print("Auto Balancing Case Bridge shutdown complete")


# 메인 실행 예제
if __name__ == "__main__":
    # 설정
    config = AutoBalancingCaseConfig(
        model_path="path/to/your/rsl_rl_checkpoint.pt",  # 실제 경로로 변경
        control_frequency=50.0,
        device="cpu",
        motor_ids=[1, 2],  # 듀얼 모터
        arduino_port="/dev/ttyACM0",  # Arduino 포트
        obs_history_length=4,
        max_episode_steps=1600
    )
    
    # Bridge 초기화
    bridge = AutoBalancingCaseBridge(config)
    
    try:
        # Load cell 캘리브레이션 (처음 한 번만)
        # bridge.calibrate_load_cells()
        
        # 한 에피소드 실행
        bridge.run_episode()
        
        # 또는 연속 실행
        # bridge.run_continuous()
    
    finally:
        bridge.shutdown()
