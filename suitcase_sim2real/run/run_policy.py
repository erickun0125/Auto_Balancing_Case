#!/usr/bin/env python3
"""
Auto Balancing Case Sim2Real Policy Runner
Isaac Lab에서 학습한 RL Policy를 실제 Auto Balancing Case 하드웨어에서 실행하는 메인 스크립트
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

from auto_balancing_case_bridge import AutoBalancingCaseBridge, AutoBalancingCaseConfig

def load_config(config_path: str) -> AutoBalancingCaseConfig:
    """YAML 설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # AutoBalancingCaseConfig 생성
    config = AutoBalancingCaseConfig(
        # Policy 설정
        model_path=config_dict['policy']['model_path'],
        control_frequency=config_dict['policy'].get('control_frequency', 50.0),
        device=config_dict['policy'].get('device', 'cpu'),
        
        # Hardware 설정 - Dual Motors
        motor_ids=config_dict['hardware']['motor']['ids'],
        motor_device=config_dict['hardware']['motor']['device'],
        motor_baudrate=config_dict['hardware']['motor']['baudrate'],
        
        # Load cell 설정 - Arduino
        arduino_port=config_dict['hardware']['arduino']['port'],
        arduino_baudrate=config_dict['hardware']['arduino']['baudrate'],
        
        # Observation 설정
        obs_history_length=config_dict['observation']['history_length'],
        
        # 정규화 파라미터
        max_wheel_force=config_dict['normalization']['max_wheel_force'],
        max_handle_force=config_dict['normalization']['max_handle_force'],
        max_joint_angle=config_dict['normalization']['max_joint_angle'],
        max_joint_velocity=config_dict['normalization']['max_joint_velocity'],
        
        # Safety 설정
        max_episode_steps=config_dict['safety']['max_episode_steps'],
        emergency_angle_limit=config_dict['safety']['emergency_angle_limit']
    )
    
    return config

def create_default_config(config_path: str):
    """기본 설정 파일 생성"""
    default_config = {
        'policy': {
            'model_path': '/path/to/your/rsl_rl_checkpoint.pt',
            'control_frequency': 50.0,
            'device': 'cpu'
        },
        'hardware': {
            'motor': {
                'ids': [1, 2],  # 듀얼 모터 ID
                'device': 'COM7',  # 모터 포트
                'baudrate': 57600
            },
            'arduino': {
                'port': 'COM7',  # Arduino 포트
                'baudrate': 115200
            }
        },
        'observation': {
            'history_length': 4
        },
        'normalization': {
            'max_wheel_force': 50.0,
            'max_handle_force': 20.0,
            'max_joint_angle': 0.5,
            'max_joint_velocity': 6.0
        },
        'safety': {
            'max_episode_steps': 400,  # IsaacLab과 일치 (50Hz * 8초)
            'emergency_angle_limit': 0.4
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"기본 설정 파일이 생성되었습니다: {config_path}")
    print("실제 하드웨어에 맞게 설정을 수정한 후 다시 실행하세요.")

def main():
    parser = argparse.ArgumentParser(description='Auto Balancing Case Sim2Real Policy Runner')
    parser.add_argument('--config', '-c', type=str, default='config/policy_config.yml',
                        help='설정 파일 경로')
    parser.add_argument('--mode', '-m', type=str, choices=['episode', 'continuous', 'calibrate'], 
                        default='calibrate', # 기본값을 calibrate로 변경 다음에 episode로 바꾸기
                        help='실행 모드: episode(한 에피소드), continuous(연속), calibrate(캘리브레이션)')
    parser.add_argument('--create-config', action='store_true',
                        help='기본 설정 파일 생성')
    
    args = parser.parse_args()
    
    # 설정 파일 경로 설정
    config_dir = Path(__file__).parent / 'config'
    config_path = config_dir / 'policy_config.yml' if args.config == 'config/policy_config.yml' else args.config
    
    # 기본 설정 파일 생성
    if args.create_config:
        config_dir.mkdir(exist_ok=True)
        create_default_config(str(config_path))
        return
    
    # 설정 파일 로드
    try:
        config = load_config(str(config_path))
        print(f"설정 파일 로드 완료: {config_path}")
    except FileNotFoundError:
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        print("다음 명령으로 기본 설정 파일을 생성하세요:")
        print(f"python {sys.argv[0]} --create-config")
        return
    except Exception as e:
        print(f"설정 파일 로드 오류: {e}")
        return
    
    # Policy 파일 존재 확인
    if not os.path.exists(config.model_path):
        print(f"Policy 파일을 찾을 수 없습니다: {config.model_path}")
        print("설정 파일에서 model_path를 확인하세요.")
        return
    
    # Bridge 초기화
    try:
        print("Auto Balancing Case Bridge 초기화 중...")
        bridge = AutoBalancingCaseBridge(config)
        print("Bridge 초기화 완료!")
    except Exception as e:
        print(f"Bridge 초기화 오류: {e}")
        return
    
    try:
        # 실행 모드에 따라 분기
        if args.mode == 'calibrate':
            print("=== Load Cell 캘리브레이션 모드 ===")
            print("주의: 캘리브레이션 중에는 지시에 따라 추를 올리고 내려야 합니다.")
            input("계속하려면 Enter를 누르세요...")
            bridge.calibrate_load_cells()
            
        elif args.mode == 'episode':
            print("=== 에피소드 모드 ===")
            print(f"최대 {config.max_episode_steps} 스텝 실행 예정")
            print("Ctrl+C로 언제든 중단할 수 있습니다.")
            input("시작하려면 Enter를 누르세요...")
            bridge.run_episode()
            
        elif args.mode == 'continuous':
            print("=== 연속 실행 모드 ===")
            print("Ctrl+C로 중단할 때까지 계속 실행됩니다.")
            input("시작하려면 Enter를 누르세요...")
            bridge.run_continuous()
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
    
    finally:
        print("정리 중...")
        bridge.shutdown()
        print("완료!")

if __name__ == "__main__":
    main()
