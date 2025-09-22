#!/usr/bin/env python3
"""
Auto Balancing Case Sim2Real Policy Runner
Isaac Lab에서 학습한 RL Policy를 실제 Auto Balancing Case 하드웨어에서 실행하는 메인 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

from auto_balancing_case_bridge import AutoBalancingCaseBridge
from config_manager import ConfigManager

def validate_config_file(config_path: str) -> bool:
    """설정 파일 존재 여부 확인"""
    if not os.path.exists(config_path):
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        print(f"기본 설정 파일이 다음 위치에 있는지 확인하세요: config/interface_config.yml")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Auto Balancing Case Sim2Real Policy Runner')
    parser.add_argument('--config', '-c', type=str, default='config/interface_config.yml',
                        help='설정 파일 경로 (기본값: config/interface_config.yml)')
    parser.add_argument('--mode', '-m', type=str, choices=['episode', 'continuous', 'calibrate', 'demo'], 
                        default='calibrate',
                        help='실행 모드: episode(한 에피소드), continuous(연속), calibrate(캘리브레이션)')
    
    args = parser.parse_args()
    
    # 설정 파일 경로 설정
    config_dir = Path(__file__).parent / 'config'
    config_path = config_dir / 'interface_config.yml' if args.config == 'config/interface_config.yml' else Path(args.config)
    
    # 설정 파일 존재 확인
    if not validate_config_file(str(config_path)):
        return
    
    # ConfigManager로 설정 로드
    try:
        print(f"설정 파일 로드 중: {config_path}")
        config_manager = ConfigManager(str(config_path))
        
        # 설정 유효성 검사
        if not config_manager.validate_config():
            print("설정 파일 유효성 검사 실패")
            return
            
        print("설정 파일 로드 완료")
        print("Config Summary:", config_manager.get_config_summary())
        
    except Exception as e:
        print(f"설정 파일 로드 오류: {e}")
        return
    
    # Policy 파일 존재 확인
    if not os.path.exists(config_manager.policy.model_path):
        print(f"Policy 파일을 찾을 수 없습니다: {config_manager.policy.model_path}")
        print("설정 파일에서 model_path를 확인하세요.")
        return
    
    # Bridge 초기화
    try:
        print("Auto Balancing Case Bridge 초기화 중...")
        bridge = AutoBalancingCaseBridge(str(config_path))
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
            print(f"최대 {config_manager.system.max_episode_steps} 스텝 실행 예정")
            print("Ctrl+C로 언제든 중단할 수 있습니다.")
            input("시작하려면 Enter를 누르세요...")
            bridge.run_episode()
            
        elif args.mode == 'continuous':
            print("=== 연속 실행 모드 ===")
            print("Ctrl+C로 중단할 때까지 계속 실행됩니다.")
            input("시작하려면 Enter를 누르세요...")
            bridge.run_continuous()

        elif args.mode == 'demo':
            print("=== 데모 모드 ===")
            input("시작하려면 Enter를 누르세요...")
            bridge.demo()
    
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
