#!/usr/bin/env python3
"""
HX711 Load Cell Interface for Auto Balancing Case (Arduino-based)
Arduino를 통해 바퀴 4개와 손잡이의 load cell 데이터를 읽는 인터페이스
"""

import numpy as np
import time
import threading
import json
from typing import Dict, List, Optional, Tuple
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    print("Warning: pyserial not installed. Install with: pip install pyserial")
    print("Mock mode enabled for development without Arduino hardware")
    SERIAL_AVAILABLE = False
    
    # Mock class for development without hardware
    class serial:
        class Serial:
            def __init__(self, port, baudrate, timeout=1): 
                self.is_open = True
            def close(self): pass
            def readline(self): return b'{"wheel_FR":0.0,"wheel_RR":0.0,"wheel_FL":0.0,"wheel_RL":0.0,"handle_1":0.0,"handle_2":0.0}\n'
            def write(self, data): pass
            def flush(self): pass

class HX711LoadCellInterface:
    """Arduino를 통해 HX711 load cell 데이터를 실시간으로 읽는 인터페이스"""
    
    def __init__(self, arduino_port: str = 'COM7', baudrate: int = 115200):
        """
        Args:
            arduino_port: Arduino 시리얼 포트 (Linux: /dev/ttyACM0, Windows: COM3)
            baudrate: 시리얼 통신 속도
        """
        self.arduino_port = arduino_port
        self.baudrate = baudrate
        
        # 시리얼 연결 초기화
        self.serial_connection = None
        self._initialize_serial_connection()
        
        # 현재 읽기 값들 (Arduino에서 받을 load cell 이름들)
        self.load_cell_names = ['wheel_FR', 'wheel_RR', 'wheel_FL', 'wheel_RL', 'handle_1', 'handle_2']
        self.current_forces = {name: 0.0 for name in self.load_cell_names}
        self.calibrated_forces = {name: 0.0 for name in self.load_cell_names}
        
        # 실시간 읽기를 위한 변수들
        self.is_reading = False
        self.read_thread = None
        self.read_frequency = 50  # Hz
        
        # 캘리브레이션을 위한 변수들
        self.zero_offsets = {name: 0.0 for name in self.load_cell_names}
        self.calibration_factors = {name: 1.0 for name in self.load_cell_names}
        
    def _initialize_serial_connection(self):
        """Arduino와의 시리얼 연결 초기화"""
        if not SERIAL_AVAILABLE:
            print("Mock mode: Creating mock serial connection")
            self.serial_connection = serial.Serial(self.arduino_port, self.baudrate)
            print("Mock Arduino 연결 완료")
            return
        
        try:
            self.serial_connection = serial.Serial(
                port=self.arduino_port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Arduino 초기화 대기
            print(f"Arduino 연결 성공: {self.arduino_port} @ {self.baudrate} baud")
            
        except Exception as e:
            print(f"Arduino 연결 실패: {e}")
            print("Mock mode로 전환합니다.")
            self.serial_connection = serial.Serial(self.arduino_port, self.baudrate)  # Mock
    
    def calibrate_all_load_cells(self, known_weight: float = 1000.0):
        """Arduino를 통해 모든 load cell 캘리브레이션 수행
        
        Args:
            known_weight: 알려진 무게 (그램 단위)
        """
        print("Load cell 캘리브레이션 시작...")
        print("모든 load cell에 무게를 제거하고 5초 기다려주세요...")
        time.sleep(5)
        
        # Arduino에 영점 조정 명령 전송
        self._send_command("TARE_ALL")
        time.sleep(2)
        
        # 영점 값 초기화
        for name in self.load_cell_names:
            self.zero_offsets[name] = 0.0
        print("모든 load cell 영점 조정 완료")
        
        print(f"\n이제 각 load cell에 {known_weight}g의 무게를 올려주세요...")
        input("준비되면 Enter를 눌러주세요...")
        
        # 캘리브레이션 값 측정
        print("캘리브레이션 값 측정 중...")
        readings = {name: [] for name in self.load_cell_names}
        
        for _ in range(10):  # 10회 측정
            data = self._read_sensor_data()
            if data:
                for name in self.load_cell_names:
                    if name in data:
                        readings[name].append(data[name])
            time.sleep(0.1)
        
        # 캘리브레이션 계수 계산
        for name in self.load_cell_names:
            if readings[name]:
                avg_reading = np.mean(readings[name])
                if avg_reading != 0:
                    self.calibration_factors[name] = avg_reading / known_weight
                    print(f"'{name}' 캘리브레이션 계수: {self.calibration_factors[name]:.6f}")
                else:
                    print(f"'{name}' 캘리브레이션 실패 - 0 값")
            else:
                print(f"'{name}' 캘리브레이션 실패 - 데이터 없음")
        
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
            
            # Arduino에서 센서 데이터 읽기
            try:
                data = self._read_sensor_data()
                if data:
                    # raw 값 업데이트
                    for name in self.load_cell_names:
                        if name in data:
                            raw_value = data[name]
                            self.current_forces[name] = raw_value
                            
                            # 캘리브레이션 적용
                            if self.calibration_factors[name] != 0:
                                calibrated_value = (raw_value - self.zero_offsets[name]) / self.calibration_factors[name]
                                self.calibrated_forces[name] = calibrated_value
                            else:
                                self.calibrated_forces[name] = 0.0
                        else:
                            self.current_forces[name] = 0.0
                            self.calibrated_forces[name] = 0.0
                            
            except Exception as e:
                print(f"Arduino 데이터 읽기 오류: {e}")
                # 오류 시 모든 값을 0으로 설정
                for name in self.load_cell_names:
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
    ################################################################################################################################################
    ################################################################################################################################################
    def get_handle_force(self) -> float:
        """손잡이의 force 값 반환"""
        handle_force = self.calibrated_forces['handle_1'] - (self.calibrated_forces['handle_1'] + self.calibrated_forces['handle_2']) / 2.0
        return handle_force
    ################################################################################################################################################
    ################################################################################################################################################
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
    
    def _send_command(self, command: str):
        """Arduino에 명령 전송"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(f"{command}\n".encode())
                self.serial_connection.flush()
            except Exception as e:
                print(f"Arduino 명령 전송 오류: {e}")
    
    def _read_sensor_data(self) -> Optional[Dict[str, float]]:
        """Arduino에서 센서 데이터 읽기"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        try:
            line = self.serial_connection.readline().decode().strip()
            if line:
                # JSON 형태로 데이터 파싱
                # 예상 형태: {"wheel_FR":123.45,"wheel_RR":67.89,"wheel_FL":234.56,"wheel_RL":78.90,"handle":12.34}
                data = json.loads(line)
                return data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Arduino 데이터 파싱 오류: {e}")
        except Exception as e:
            print(f"Arduino 데이터 읽기 오류: {e}")
        
        return None
    
    def shutdown(self):
        """정리 및 종료"""
        self.stop_real_time_reading()
        
        # 시리얼 연결 종료
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        print("HX711 Load Cell Arduino 인터페이스 종료")


# 사용 예제
if __name__ == "__main__":
    # Arduino 기반 Load cell 인터페이스 초기화
    load_cell_interface = HX711LoadCellInterface(
        arduino_port='/dev/ttyACM0',  # Linux
        # arduino_port='COM3',        # Windows
        baudrate=115200
    )
    
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
