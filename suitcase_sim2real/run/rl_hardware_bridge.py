#!/usr/bin/env python3
"""
Isaac Lab에서 학습한 RL Policy를 실제 Dynamixel 하드웨어로 실행하는 브릿지
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pickle
import yaml

from dynamixel_hardware_interface import DynamixelXL430Interface

@dataclass
class PolicyConfig:
    """Policy 설정"""
    model_path: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    control_frequency: float = 50.0  # Hz
    device: str = "cpu"

class RLHardwareBridge:
    """RL Policy와 실제 하드웨어 간의 브릿지"""
    
    def __init__(self, policy_config: PolicyConfig, motor_ids: list):
        self.config = policy_config
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        
        # Policy 로드
        self.policy = self._load_policy()
        
        # Hardware interface 초기화
        self.hw_interface = DynamixelXL430Interface(
            motor_ids=motor_ids,
            control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
        )
        
        # 제어 주기 설정
        self.control_dt = 1.0 / policy_config.control_frequency
        
        # 상태 기록을 위한 변수들
        self.episode_step = 0
        self.max_episode_steps = 1000
        self.is_running = False
        
        # Observation history (일부 policy는 이전 관측이 필요)
        self.obs_history_length = 1  # Isaac Lab policy에 따라 조정
        self.obs_history = []
        
    def _load_policy(self) -> torch.nn.Module:
        """Isaac Lab에서 학습된 policy 모델 로드"""
        print(f"Loading policy from {self.config.model_path}")
        
        # PyTorch 모델 로드 (Isaac Lab 표준 방식)
        try:
            # 방법 1: torch.jit으로 저장된 경우
            policy = torch.jit.load(self.config.model_path, map_location=self.config.device)
        except:
            try:
                # 방법 2: state_dict로 저장된 경우
                checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
                # 여기서 실제 모델 아키텍처를 다시 생성해야 함
                # Isaac Lab의 경우 보통 MLP 정책
                policy = self._create_policy_network(checkpoint)
            except:
                # 방법 3: pickle로 저장된 경우
                with open(self.config.model_path, 'rb') as f:
                    policy = pickle.load(f)
        
        policy.eval()  # 평가 모드
        print("Policy loaded successfully")
        return policy
    
    def _create_policy_network(self, checkpoint: Dict) -> torch.nn.Module:
        """Policy 네트워크 구조 생성 (Isaac Lab MLP 기본 구조)"""
        # Isaac Lab에서 일반적으로 사용하는 MLP 구조
        import torch.nn as nn
        
        # observation과 action dimension 계산
        obs_dim = sum([np.prod(space['shape']) for space in self.config.observation_space.values()])
        action_dim = np.prod(self.config.action_space['shape'])
        
        class MLPPolicy(nn.Module):
            def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
                super().__init__()
                layers = []
                in_dim = obs_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU()
                    ])
                    in_dim = hidden_dim
                
                layers.append(nn.Linear(in_dim, action_dim))
                layers.append(nn.Tanh())  # 일반적으로 action은 -1~1로 정규화
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, obs):
                return self.network(obs)
        
        policy = MLPPolicy(obs_dim, action_dim)
        policy.load_state_dict(checkpoint['model_state_dict'])
        return policy
    
    def _prepare_observation(self, raw_obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """하드웨어에서 받은 observation을 policy 입력 형태로 변환"""
        # Isaac Lab에서 일반적으로 사용하는 observation 구조에 맞춰 변환
        obs_list = []
        
        # observation space 순서에 맞춰 concatenate
        for key in sorted(self.config.observation_space.keys()):
            if key in raw_obs:
                obs_value = raw_obs[key]
                # flatten if needed
                if obs_value.ndim > 1:
                    obs_value = obs_value.flatten()
                obs_list.append(obs_value)
        
        # numpy to torch tensor
        obs_tensor = torch.from_numpy(np.concatenate(obs_list)).float()
        
        # Add batch dimension
        obs_tensor = obs_tensor.unsqueeze(0)
        
        return obs_tensor.to(self.config.device)
    
    def _process_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        """Policy 출력을 하드웨어 명령으로 변환"""
        # Remove batch dimension and convert to numpy
        action = action_tensor.squeeze(0).detach().cpu().numpy()
        
        # Isaac Lab에서는 보통 action이 -1~1로 정규화되어 있음
        # 이를 실제 모터 명령으로 변환
        motor_commands = self.hw_interface.denormalize_actions(action)
        
        return motor_commands
    
    def run_episode(self):
        """한 에피소드 실행"""
        print("Starting episode...")
        
        # Hardware interface 시작
        self.hw_interface.start_real_time_reading()
        
        # 초기 위치로 이동
        initial_positions = np.array([2048] * self.num_motors)  # 중간 위치
        self.hw_interface.set_position_commands(initial_positions)
        time.sleep(2.0)  # 초기 위치 도달 대기
        
        self.episode_step = 0
        self.is_running = True
        
        try:
            while self.is_running and self.episode_step < self.max_episode_steps:
                step_start_time = time.time()
                
                # 1. Hardware에서 현재 상태 읽기
                raw_obs = self.hw_interface.get_observations()
                normalized_obs = self.hw_interface.normalize_observations(raw_obs)
                
                # 2. Observation 준비
                policy_obs = self._prepare_observation(normalized_obs)
                
                # 3. Policy 실행
                with torch.no_grad():
                    action = self.policy(policy_obs)
                
                # 4. Action을 motor command로 변환
                motor_commands = self._process_action(action)
                
                # 5. Hardware에 명령 전송
                self.hw_interface.set_position_commands(motor_commands)
                
                # 6. 디버깅 정보 출력
                if self.episode_step % 10 == 0:  # 10스텝마다 출력
                    print(f"Step {self.episode_step}:")
                    print(f"  Current pos: {raw_obs['joint_pos'][:4]}")  # 처음 4개만
                    print(f"  Target pos:  {motor_commands[:4]}")
                    print(f"  Action:      {action.squeeze().cpu().numpy()[:4]}")
                
                # 7. 제어 주기 유지
                elapsed = time.time() - step_start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:  # 10ms 이상 지연 시 경고
                    print(f"Warning: Control loop delay: {-sleep_time:.3f}s")
                
                self.episode_step += 1
        
        except KeyboardInterrupt:
            print("Episode interrupted by user")
        
        finally:
            print("Episode finished")
            self.hw_interface.stop_real_time_reading()
    
    def run_continuous(self):
        """연속 실행 (에피소드 개념 없이)"""
        print("Starting continuous control...")
        
        self.hw_interface.start_real_time_reading()
        
        # 초기화
        initial_positions = np.array([2048] * self.num_motors)
        self.hw_interface.set_position_commands(initial_positions)
        time.sleep(2.0)
        
        step_count = 0
        
        try:
            while True:
                step_start_time = time.time()
                
                # Hardware 상태 읽기
                raw_obs = self.hw_interface.get_observations()
                normalized_obs = self.hw_interface.normalize_observations(raw_obs)
                
                # Policy 실행
                policy_obs = self._prepare_observation(normalized_obs)
                with torch.no_grad():
                    action = self.policy(policy_obs)
                
                # 명령 전송
                motor_commands = self._process_action(action)
                self.hw_interface.set_position_commands(motor_commands)
                
                # 모니터링
                if step_count % 50 == 0:  # 1초마다 출력 (50Hz 기준)
                    print(f"Step {step_count}: Pos={raw_obs['joint_pos'][:2]}, "
                          f"Cmd={motor_commands[:2]}")
                
                # 제어 주기 유지
                elapsed = time.time() - step_start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                step_count += 1
        
        except KeyboardInterrupt:
            print("Continuous control stopped")
        
        finally:
            self.hw_interface.stop_real_time_reading()
    
    def shutdown(self):
        """정리 및 종료"""
        self.is_running = False
        self.hw_interface.shutdown()


# 메인 실행 예제
if __name__ == "__main__":
    # Policy 설정 (실제 Isaac Lab 학습 결과에 맞춰 수정)
    policy_config = PolicyConfig(
        model_path="path/to/your/trained_policy.pt",  # 실제 경로로 변경
        observation_space={
            'joint_pos': {'shape': [4]},        # 4개 관절 위치
            'joint_vel': {'shape': [4]},        # 4개 관절 속도
            'joint_effort': {'shape': [4]},     # 4개 관절 토크
            'joint_pos_target': {'shape': [4]}  # 4개 관절 목표 위치
        },
        action_space={'shape': [4]},  # 4개 관절에 대한 action
        control_frequency=50.0,  # 50Hz 제어
        device="cpu"  # 또는 "cuda"
    )
    
    # 모터 ID 설정
    motor_ids = [1, 2, 3, 4]
    
    # Bridge 초기화
    bridge = RLHardwareBridge(policy_config, motor_ids)
    
    try:
        # 한 에피소드 실행
        bridge.run_episode()
        
        # 또는 연속 실행
        # bridge.run_continuous()
    
    finally:
        bridge.shutdown()