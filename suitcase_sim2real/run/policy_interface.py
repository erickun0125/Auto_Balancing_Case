#!/usr/bin/env python3
"""
Policy Interface for RSL-RL ActorCritic Models
Isaac Lab에서 학습한 RSL-RL Policy를 로드하고 inference를 수행하는 인터페이스
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    """Policy 설정"""
    model_path: str
    device: str = "cpu"
    obs_history_length: int = 4
    
    # RSL-RL ActorCritic 네트워크 설정 (실제 학습된 모델과 일치해야 함)
    actor_hidden_dims: Optional[List[int]] = None
    critic_hidden_dims: Optional[List[int]] = None
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    
    # Observation dimensions (IsaacLab 환경에 맞춰 설정)
    obs_dim_per_timestep: int = 8  # joint_pos(1) + joint_vel(1) + prev_action(1) + wheel_forces(4) + handle_force(1)
    num_actions: int = 1  # 단일 밸런싱 조인트
    
    def __post_init__(self):
        # 기본값 설정
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 128, 64]  # rsl_rl_ppo_cfg.py에서 확인된 실제 구조
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 128, 64]


class RSLRLActorCritic(nn.Module):
    """RSL-RL ActorCritic 네트워크 (원본 구조와 동일)"""
    
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        
        # Observation dimensions 계산
        num_actor_obs = config.obs_dim_per_timestep * config.obs_history_length
        num_critic_obs = num_actor_obs
        num_actions = config.num_actions
        
        # Hidden dimensions 확인 및 기본값 설정
        actor_hidden_dims = config.actor_hidden_dims or [256, 128, 64]
        critic_hidden_dims = config.critic_hidden_dims or [256, 128, 64]
        
        # Activation function 설정
        activation = self._get_activation(config.activation)
        
        # Actor network (RSL-RL 구조와 동일)
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (RSL-RL 구조와 동일)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise (RSL-RL 구조와 동일)
        self.noise_std_type = config.noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(config.init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(config.init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    def _get_activation(self, activation_name: str):
        """Activation function 반환"""
        if activation_name.lower() == "elu":
            return nn.ELU()
        elif activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "tanh":
            return nn.Tanh()
        else:
            return nn.ELU()  # 기본값
    
    def update_distribution(self, observations):
        """RSL-RL과 동일한 distribution 업데이트"""
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)
    
    def act_inference(self, observations):
        """Inference 모드에서 action 생성 (노이즈 없음)"""
        return self.actor(observations)
    
    def forward(self, observations):
        """Forward pass - inference 모드로 사용"""
        return self.act_inference(observations)


class PolicyInterface:
    """RSL-RL Policy 로드 및 inference 인터페이스"""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.policy = None
        self._load_policy()
    
    def _load_policy(self):
        """Isaac Lab RSL-RL policy 로드"""
        print(f"Loading policy from {self.config.model_path}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Policy file not found: {self.config.model_path}")
        
        try:
            # RSL-RL checkpoint 로드
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
            
            # RSL-RL OnPolicyRunner checkpoint 구조 확인
            print("Checkpoint keys:", list(checkpoint.keys()))
            
            # RSL-RL OnPolicyRunner에서는 'model_state_dict'가 ActorCritic 전체를 포함
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                # 직접 state_dict인 경우
                model_state_dict = checkpoint
            
            # ActorCritic 모델 생성
            self.policy = RSLRLActorCritic(self.config)
            
            # 전체 모델 로드
            self.policy.load_state_dict(model_state_dict, strict=True)
            self.policy.eval()
            
            print("RSL-RL ActorCritic model loaded successfully")
            print("Model structure:")
            print(f"  Actor: {self.policy.actor}")
            print(f"  Critic: {self.policy.critic}")
            print(f"  Total parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
            
        except Exception as e:
            print(f"Policy loading error: {e}")
            if 'checkpoint' in locals():
                print("Available checkpoint keys:", list(checkpoint.keys()))
            raise
    
    def prepare_policy_input(self, obs_history: List[Dict[str, np.ndarray]]) -> torch.Tensor:
        """Observation history를 policy 입력으로 변환"""
        if len(obs_history) == 0:
            raise ValueError("Observation history is empty")
        
        # History를 concatenate (가장 오래된 것부터 최신 순서로)
        obs_vectors = []
        for obs in obs_history:
            # 각 observation을 벡터로 변환 (IsaacLab 환경과 동일한 순서)
            obs_vector = np.concatenate([
                obs['joint_pos'],              # 1차원 - joint position
                obs['joint_vel'],              # 1차원 - joint velocity  
                obs['prev_action'],            # 1차원 - previous action
                obs['wheel_contact_forces'],   # 4차원 - wheel contact forces
                obs['handle_external_force']   # 1차원 - handle external force
            ])  # 총 8차원 per timestep
            obs_vectors.append(obs_vector)
        
        # History가 부족한 경우 0으로 패딩 (oldest timesteps)
        while len(obs_vectors) < self.config.obs_history_length:
            obs_vectors.insert(0, np.zeros(self.config.obs_dim_per_timestep))
        
        # 최신 N개만 사용 (history_length 초과 시)
        if len(obs_vectors) > self.config.obs_history_length:
            obs_vectors = obs_vectors[-self.config.obs_history_length:]
        
        # Concatenate all history: 8차원 * 4 = 32차원
        policy_input = np.concatenate(obs_vectors)  
        
        # Convert to torch tensor with batch dimension
        policy_input = torch.from_numpy(policy_input).float().unsqueeze(0)
        
        return policy_input.to(self.config.device)
    
    def predict(self, obs_history: List[Dict[str, np.ndarray]]) -> float:
        """Policy prediction (inference)"""
        if self.policy is None:
            raise RuntimeError("Policy not loaded")
        
        # Policy 입력 준비
        policy_input = self.prepare_policy_input(obs_history)
        
        # Inference 수행
        with torch.no_grad():
            action_tensor = self.policy.act_inference(policy_input)
            action = action_tensor.squeeze().cpu().numpy()
            
            # Scalar action인 경우 처리
            if action.ndim == 0:
                action = float(action)
            else:
                action = action[0]
        
        return action
    
    def get_value(self, obs_history: List[Dict[str, np.ndarray]]) -> float:
        """Critic value prediction (디버깅용)"""
        if self.policy is None:
            raise RuntimeError("Policy not loaded")
        
        policy_input = self.prepare_policy_input(obs_history)
        
        with torch.no_grad():
            value_tensor = self.policy.critic(policy_input)
            value = value_tensor.squeeze().cpu().numpy()
            
            if value.ndim == 0:
                value = float(value)
            else:
                value = value[0]
        
        return value
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.policy is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "actor_hidden_dims": self.config.actor_hidden_dims,
            "critic_hidden_dims": self.config.critic_hidden_dims,
            "activation": self.config.activation,
            "obs_dim": self.config.obs_dim_per_timestep * self.config.obs_history_length,
            "num_actions": self.config.num_actions,
            "device": self.config.device,
            "total_parameters": sum(p.numel() for p in self.policy.parameters())
        }


class PolicyInterfaceDemo:

    def prepare_policy_input(self, obs_history: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Observation history를 policy 입력으로 변환"""
        if len(obs_history) == 0:
            raise ValueError("Observation history is empty")
        
        # theta_t: 최신 step의 joint_pos
        last_obs = obs_history[-1]
        theta_t = last_obs['joint_pos']

        # delta_f: 최근 4 step의 handle_external_force 평균 (부족하면 존재하는 만큼 평균)
        recent_obs = obs_history[-4:]
        recent_forces = np.stack([obs['handle_external_force'] for obs in recent_obs], axis=0)
        delta_f = recent_forces.mean(axis=0)

        return theta_t, delta_f
    
    def predict(self, obs_history: List[Dict[str, np.ndarray]]) -> float:
        """Policy prediction (inference)"""
        # Policy 입력 준비
        theta_t, delta_f  = self.prepare_policy_input(obs_history)
        
        k_p = 0.0001
        
        # 스칼라 보장 (형상 불일치/ndarray 반환을 방지)
        theta_t_scalar = float(np.asarray(theta_t).reshape(-1)[0])
        delta_f_scalar = float(np.asarray(delta_f).mean())
        
        action = theta_t_scalar + k_p * delta_f_scalar
        
        return action

# 테스트용 메인 함수
if __name__ == "__main__":
    # 테스트 설정
    config = PolicyConfig(
        model_path="/home/eric/custom/Auto_Balancing_Suitcase/suitcase_learning/trained_model/model_19999.pt",
        device="cpu",
        obs_history_length=4
    )
    
    # Policy 인터페이스 생성
    policy_interface = PolicyInterface(config)
    
    # 모델 정보 출력
    model_info = policy_interface.get_model_info()
    print("Model Info:", model_info)
    
    # 더미 observation으로 테스트
    dummy_obs = {
        'joint_pos': np.array([0.1]),
        'joint_vel': np.array([0.05]),
        'prev_action': np.array([0.0]),
        'wheel_contact_forces': np.array([10.0, 12.0, 11.0, 13.0]),
        'handle_external_force': np.array([2.0])
    }
    
    obs_history = [dummy_obs] * 4
    
    # Prediction 테스트
    try:
        action = policy_interface.predict(obs_history)
        value = policy_interface.get_value(obs_history)
        print(f"Test prediction - Action: {action:.4f}, Value: {value:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
