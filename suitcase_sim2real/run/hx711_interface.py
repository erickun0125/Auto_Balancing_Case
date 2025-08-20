#!/usr/bin/env python3
"""
HX711 Load Cell Interface for Auto Balancing Case
HX711을 통해 바퀴 4개와 손잡이의 load cell 데이터를 읽는 인터페이스
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
try:
    import RPi.GPIO as GPIO
    from hx711 import HX711
    HX711_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO or hx711 not installed. Install with: pip install RPi.GPIO hx711")
    print("Mock mode enabled for development without Raspberry Pi hardware")
    HX711_AVAILABLE = False
    
    # Mock classes for development without hardware
    class GPIO:
        BCM = 'BCM'
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def cleanup(): pass
    
    class HX711:
        def __init__(self, dout, pd_sck): pass
        def set_reading_format(self, f1, f2): pass
        def set_reference_unit(self, unit): pass
        def reset(self): pass
        def tare(self): pass
        def get_value(self, times=1): return 0.0
        def power_down(self): pass

class HX711LoadCellInterface:
    """HX711을 사용하여 여러 load cell의 데이터를 실시간으로 읽는 인터페이스"""
    
    def __init__(self, load_cell_configs: List[Dict]):
        """
        Args:
            load_cell_configs: load cell 설정 리스트
                예: [
                    {'name': 'wheel_FL', 'dout_pin': 5, 'pd_sck_pin': 6, 'calibration_factor': 1.0},
                    {'name': 'wheel_FR', 'dout_pin': 13, 'pd_sck_pin': 19, 'calibration_factor': 1.0},
                    {'name': 'wheel_RL', 'dout_pin': 16, 'pd_sck_pin': 20, 'calibration_factor': 1.0},
                    {'name': 'wheel_RR', 'dout_pin': 21, 'pd_sck_pin': 26, 'calibration_factor': 1.0},
                    {'name': 'handle', 'dout_pin': 12, 'pd_sck_pin': 25, 'calibration_factor': 1.0}
                ]
        """
        self.load_cell_configs = load_cell_configs
        self.num_load_cells = len(load_cell_configs)
        
        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        
        # HX711 인스턴스들 초기화
        self.hx711_instances = {}
        self._initialize_load_cells()
        
        # 현재 읽기 값들
        self.current_forces = {config['name']: 0.0 for config in load_cell_configs}
        self.calibrated_forces = {config['name']: 0.0 for config in load_cell_configs}
        
        # 실시간 읽기를 위한 변수들
        self.is_reading = False
        self.read_thread = None
        self.read_frequency = 50  # Hz
        
        # 캘리브레이션을 위한 변수들
        self.zero_offsets = {config['name']: 0.0 for config in load_cell_configs}
        self.calibration_factors = {config['name']: config.get('calibration_factor', 1.0) 
                                  for config in load_cell_configs}
        
    def _initialize_load_cells(self):
        """모든 load cell HX711 인스턴스 초기화"""
        if not HX711_AVAILABLE:
            print("Mock mode: Creating mock HX711 instances")
            for config in self.load_cell_configs:
                name = config['name']
                self.hx711_instances[name] = HX711(0, 0)  # Mock instance
                print(f"Mock load cell '{name}' 초기화 완료")
            return
        
        for config in self.load_cell_configs:
            name = config['name']
            dout_pin = config['dout_pin']
            pd_sck_pin = config['pd_sck_pin']
            
            try:
                hx = HX711(dout_pin, pd_sck_pin)
                hx.set_reading_format("MSB", "MSB")
                hx.set_reference_unit(1)  # 기본 참조 단위
                hx.reset()
                hx.tare()  # 영점 조정
                
                self.hx711_instances[name] = hx
                print(f"Load cell '{name}' 초기화 완료 (DOUT: {dout_pin}, PD_SCK: {pd_sck_pin})")
                
            except Exception as e:
                print(f"Load cell '{name}' 초기화 실패: {e}")
                self.hx711_instances[name] = None
    
    def calibrate_all_load_cells(self, known_weight: float = 1000.0):
        """모든 load cell 캘리브레이션 수행
        
        Args:
            known_weight: 알려진 무게 (그램 단위)
        """
        print("Load cell 캘리브레이션 시작...")
        print("모든 load cell에 무게를 제거하고 5초 기다려주세요...")
        time.sleep(5)
        
        # 영점 조정
        for name, hx in self.hx711_instances.items():
            if hx is not None:
                hx.tare()
                self.zero_offsets[name] = 0.0
                print(f"'{name}' 영점 조정 완료")
        
        print(f"\n이제 각 load cell에 {known_weight}g의 무게를 올려주세요...")
        input("준비되면 Enter를 눌러주세요...")
        
        # 캘리브레이션 값 측정
        for name, hx in self.hx711_instances.items():
            if hx is not None:
                print(f"'{name}' 캘리브레이션 중...")
                readings = []
                for _ in range(10):  # 10회 측정
                    val = hx.get_value(5)  # 5회 평균
                    readings.append(val)
                    time.sleep(0.1)
                
                avg_reading = np.mean(readings)
                if avg_reading != 0:
                    self.calibration_factors[name] = avg_reading / known_weight
                    print(f"'{name}' 캘리브레이션 계수: {self.calibration_factors[name]:.6f}")
                else:
                    print(f"'{name}' 캘리브레이션 실패 - 0 값")
        
        print("캘리브레이션 완료!")
    
    def start_real_time_reading(self):
        """백그라운드에서 실시간으로 load cell 데이터 읽기 시작"""
        if self.is_reading:
            return
        
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Load cell 실시간 읽기 시작")
    
    def stop_real_time_reading(self):
        """실시간 읽기 중지"""
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join()
        print("Load cell 실시간 읽기 중지")
    
    def _read_loop(self):
        """백그라운드에서 실행되는 읽기 루프"""
        while self.is_reading:
            start_time = time.time()
            
            # 모든 load cell에서 데이터 읽기
            for name, hx in self.hx711_instances.items():
                if hx is not None:
                    try:
                        # raw 값 읽기
                        raw_value = hx.get_value(1)  # 1회 측정으로 빠른 읽기
                        self.current_forces[name] = raw_value
                        
                        # 캘리브레이션 적용
                        if self.calibration_factors[name] != 0:
                            calibrated_value = (raw_value - self.zero_offsets[name]) / self.calibration_factors[name]
                            self.calibrated_forces[name] = calibrated_value
                        else:
                            self.calibrated_forces[name] = 0.0
                            
                    except Exception as e:
                        print(f"Load cell '{name}' 읽기 오류: {e}")
                        self.current_forces[name] = 0.0
                        self.calibrated_forces[name] = 0.0
            
            # 주기 유지
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_wheel_forces(self) -> np.ndarray:
        """4개 바퀴의 force 값을 numpy 배열로 반환 (IsaacLab USD 순서: FR, RR, FL, RL)"""
        wheel_names = ['wheel_FR', 'wheel_RR', 'wheel_FL', 'wheel_RL']
        forces = []
        
        for name in wheel_names:
            if name in self.calibrated_forces:
                forces.append(abs(self.calibrated_forces[name]))  # 절댓값 사용
            else:
                forces.append(0.0)
        
        return np.array(forces)
    
    def get_handle_force(self) -> float:
        """손잡이의 force 값 반환"""
        if 'handle' in self.calibrated_forces:
            return abs(self.calibrated_forces['handle'])  # 절댓값 사용
        return 0.0
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """RL Policy를 위한 observation 데이터 반환"""
        return {
            'wheel_forces': self.get_wheel_forces(),
            'handle_force': np.array([self.get_handle_force()]),
            'raw_forces': np.array(list(self.current_forces.values())),
            'calibrated_forces': np.array(list(self.calibrated_forces.values()))
        }
    
    def normalize_observations(self, obs: Dict[str, np.ndarray], 
                             max_wheel_force: float = 50.0, 
                             max_handle_force: float = 20.0) -> Dict[str, np.ndarray]:
        """Observation을 RL에서 사용하는 정규화된 범위로 변환
        
        Args:
            obs: 원본 observation
            max_wheel_force: 바퀴 힘의 최대값 (정규화용)
            max_handle_force: 손잡이 힘의 최대값 (정규화용)
        """
        normalized = {}
        
        # Wheel forces: 0~max_wheel_force -> 0~1
        normalized['wheel_forces'] = np.clip(obs['wheel_forces'] / max_wheel_force, 0.0, 1.0)
        
        # Handle force: 0~max_handle_force -> 0~1  
        normalized['handle_force'] = np.clip(obs['handle_force'] / max_handle_force, 0.0, 1.0)
        
        return normalized
    
    def save_calibration(self, filename: str = 'load_cell_calibration.npz'):
        """캘리브레이션 데이터 저장"""
        np.savez(filename,
                zero_offsets=self.zero_offsets,
                calibration_factors=self.calibration_factors)
        print(f"캘리브레이션 데이터가 {filename}에 저장되었습니다.")
    
    def load_calibration(self, filename: str = 'load_cell_calibration.npz'):
        """캘리브레이션 데이터 로드"""
        try:
            data = np.load(filename, allow_pickle=True)
            self.zero_offsets = data['zero_offsets'].item()
            self.calibration_factors = data['calibration_factors'].item()
            print(f"캘리브레이션 데이터가 {filename}에서 로드되었습니다.")
        except FileNotFoundError:
            print(f"캘리브레이션 파일 {filename}을 찾을 수 없습니다.")
        except Exception as e:
            print(f"캘리브레이션 로드 오류: {e}")
    
    def shutdown(self):
        """정리 및 종료"""
        self.stop_real_time_reading()
        
        # GPIO 정리
        for hx in self.hx711_instances.values():
            if hx is not None:
                hx.power_down()
        
        GPIO.cleanup()
        print("HX711 Load Cell 인터페이스 종료")


# 사용 예제
if __name__ == "__main__":
    # Load cell 설정 (실제 GPIO 핀 번호에 맞춰 수정 필요)
    # IsaacLab USD 파일의 바퀴 순서와 일치: FR, RR, FL, RL
    load_cell_configs = [
        {'name': 'wheel_FR', 'dout_pin': 5, 'pd_sck_pin': 6, 'calibration_factor': 1.0},
        {'name': 'wheel_RR', 'dout_pin': 13, 'pd_sck_pin': 19, 'calibration_factor': 1.0},
        {'name': 'wheel_FL', 'dout_pin': 16, 'pd_sck_pin': 20, 'calibration_factor': 1.0},
        {'name': 'wheel_RL', 'dout_pin': 21, 'pd_sck_pin': 26, 'calibration_factor': 1.0},
        {'name': 'handle', 'dout_pin': 12, 'pd_sck_pin': 25, 'calibration_factor': 1.0}
    ]
    
    # Load cell 인터페이스 초기화
    load_cell_interface = HX711LoadCellInterface(load_cell_configs)
    
    try:
        # 캘리브레이션 수행 (선택사항)
        # load_cell_interface.calibrate_all_load_cells(1000.0)  # 1kg 추 사용
        
        # 또는 저장된 캘리브레이션 로드
        load_cell_interface.load_calibration()
        
        # 실시간 읽기 시작
        load_cell_interface.start_real_time_reading()
        
        # 10초간 데이터 모니터링
        for i in range(100):
            obs = load_cell_interface.get_observations()
            normalized_obs = load_cell_interface.normalize_observations(obs)
            
            print(f"Step {i}:")
            print(f"  Wheel forces: {obs['wheel_forces']}")
            print(f"  Handle force: {obs['handle_force'][0]:.2f}")
            print(f"  Normalized wheels: {normalized_obs['wheel_forces']}")
            time.sleep(0.1)
    
    finally:
        load_cell_interface.shutdown()
