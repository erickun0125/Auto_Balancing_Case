#!/usr/bin/env python3
"""
Dynamixel XL430 Hardware Interface for RL Policy Integration
XL430은 X-Series Protocol 2.0을 사용
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
from dynamixel_sdk import *

class DynamixelXL430Interface:
    """XL430 모터들과 실시간 통신하는 하드웨어 인터페이스"""
    
    # XL430 Control Table Addresses (Protocol 2.0)
    ADDR_OPERATING_MODE = 11
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_GOAL_VELOCITY = 104
    ADDR_GOAL_CURRENT = 102
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_CURRENT = 126
    ADDR_HARDWARE_ERROR_STATUS = 70
    
    # XL430 Specifications
    POSITION_RANGE = 4095  # 0-4095 (12-bit)
    VELOCITY_RANGE = 1023  # 0-1023
    CURRENT_RANGE = 1193   # -1193 to 1193
    
    # Control Modes
    CURRENT_CONTROL_MODE = 0
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4
    
    def __init__(self, motor_ids: List[int], device_name: str = '/dev/ttyUSB0', 
                 baudrate: int = 57600, control_mode: int = POSITION_CONTROL_MODE):
        """
        Args:
            motor_ids: 모터 ID 리스트 (예: [1, 2, 3, 4])
            device_name: 시리얼 포트 (Linux: /dev/ttyUSB0, Windows: COM3)
            baudrate: 통신 속도
            control_mode: 제어 모드 (위치/속도/전류)
        """
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        self.control_mode = control_mode
        
        # SDK 초기화
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(2.0)  # Protocol 2.0
        
        # 상태 저장 변수들
        self.current_positions = np.zeros(self.num_motors)
        self.current_velocities = np.zeros(self.num_motors)
        self.current_currents = np.zeros(self.num_motors)
        self.goal_positions = np.zeros(self.num_motors)
        
        # 실시간 읽기를 위한 변수들
        self.is_reading = False
        self.read_thread = None
        self.read_frequency = 100  # Hz
        
        self._initialize_connection()
        self._setup_motors()
    
    def _initialize_connection(self):
        """시리얼 연결 초기화"""
        if not self.port_handler.openPort():
            raise RuntimeError("포트 열기 실패")
        
        if not self.port_handler.setBaudRate(57600):
            raise RuntimeError("Baudrate 설정 실패")
        
        print("Dynamixel 연결 성공")
    
    def _setup_motors(self):
        """모터들 초기 설정"""
        for motor_id in self.motor_ids:
            # 토크 비활성화
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0)
            
            # 제어 모드 설정
            self._write_1byte(motor_id, self.ADDR_OPERATING_MODE, self.control_mode)
            
            # 토크 활성화
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 1)
            
            print(f"Motor {motor_id} 초기화 완료")
    
    def _write_1byte(self, motor_id: int, address: int, value: int):
        """1바이트 데이터 쓰기"""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, address, value)
        if result != COMM_SUCCESS:
            print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            print(f"Hardware error: {self.packet_handler.getRxPacketError(error)}")
    
    def _write_4byte(self, motor_id: int, address: int, value: int):
        """4바이트 데이터 쓰기"""
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, address, value)
        if result != COMM_SUCCESS:
            print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            print(f"Hardware error: {self.packet_handler.getRxPacketError(error)}")
    
    def _read_4byte(self, motor_id: int, address: int) -> int:
        """4바이트 데이터 읽기"""
        value, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, address)
        if result != COMM_SUCCESS:
            print(f"Read error: {self.packet_handler.getTxRxResult(result)}")
            return 0
        if error != 0:
            print(f"Hardware error: {self.packet_handler.getRxPacketError(error)}")
        return value
    
    def start_real_time_reading(self):
        """백그라운드에서 실시간으로 센서 데이터 읽기 시작"""
        if self.is_reading:
            return
        
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("실시간 읽기 시작")
    
    def stop_real_time_reading(self):
        """실시간 읽기 중지"""
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join()
        print("실시간 읽기 중지")
    
    def _read_loop(self):
        """백그라운드에서 실행되는 읽기 루프"""
        while self.is_reading:
            start_time = time.time()
            
            # 모든 모터의 현재 상태 읽기
            for i, motor_id in enumerate(self.motor_ids):
                # Position 읽기 (4바이트, 0-4095)
                pos = self._read_4byte(motor_id, self.ADDR_PRESENT_POSITION)
                self.current_positions[i] = pos
                
                # Velocity 읽기 (4바이트, signed)
                vel = self._read_4byte(motor_id, self.ADDR_PRESENT_VELOCITY)
                # 부호 있는 32비트로 변환
                if vel > 2147483647:
                    vel -= 4294967296
                self.current_velocities[i] = vel
                
                # Current 읽기 (2바이트, signed)
                curr = self._read_4byte(motor_id, self.ADDR_PRESENT_CURRENT)
                if curr > 32767:
                    curr -= 65536
                self.current_currents[i] = curr
            
            # 주기 유지
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def set_position_commands(self, positions: np.ndarray):
        """위치 명령 전송 (RL Policy의 action을 모터 명령으로 변환)"""
        if len(positions) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} positions, got {len(positions)}")
        
        for i, motor_id in enumerate(self.motor_ids):
            # Position 범위 확인 및 클리핑
            pos_cmd = int(np.clip(positions[i], 0, self.POSITION_RANGE))
            self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, pos_cmd)
            self.goal_positions[i] = pos_cmd
    
    def set_velocity_commands(self, velocities: np.ndarray):
        """속도 명령 전송"""
        if self.control_mode != self.VELOCITY_CONTROL_MODE:
            print("Warning: 속도 제어 모드가 아닙니다")
        
        for i, motor_id in enumerate(self.motor_ids):
            # Velocity 범위 확인 및 클리핑
            vel_cmd = int(np.clip(velocities[i], 0, self.VELOCITY_RANGE))
            self._write_4byte(motor_id, self.ADDR_GOAL_VELOCITY, vel_cmd)
    
    def set_current_commands(self, currents: np.ndarray):
        """전류 명령 전송"""
        if self.control_mode != self.CURRENT_CONTROL_MODE:
            print("Warning: 전류 제어 모드가 아닙니다")
        
        for i, motor_id in enumerate(self.motor_ids):
            # Current 범위 확인 및 클리핑
            curr_cmd = int(np.clip(currents[i], -self.CURRENT_RANGE, self.CURRENT_RANGE))
            if curr_cmd < 0:
                curr_cmd += 65536  # 2의 보수
            self._write_4byte(motor_id, self.ADDR_GOAL_CURRENT, curr_cmd)
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """RL Policy를 위한 observation 데이터 반환"""
        return {
            'joint_pos': self.current_positions.copy(),
            'joint_vel': self.current_velocities.copy(), 
            'joint_effort': self.current_currents.copy(),
            'joint_pos_target': self.goal_positions.copy()
        }
    
    def normalize_observations(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Observation을 RL에서 사용하는 정규화된 범위로 변환"""
        normalized = {}
        
        # Position: 0-4095 -> -1 to 1
        normalized['joint_pos'] = (obs['joint_pos'] / (self.POSITION_RANGE/2)) - 1.0
        
        # Velocity: raw velocity -> normalized (최대속도 기준)
        max_vel = 1023  # XL430 최대 속도
        normalized['joint_vel'] = np.clip(obs['joint_vel'] / max_vel, -1.0, 1.0)
        
        # Current: -1193~1193 -> -1 to 1  
        normalized['joint_effort'] = obs['joint_effort'] / self.CURRENT_RANGE
        
        # Target position
        normalized['joint_pos_target'] = (obs['joint_pos_target'] / (self.POSITION_RANGE/2)) - 1.0
        
        return normalized
    
    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """RL Policy의 정규화된 action을 실제 모터 명령으로 변환"""
        if self.control_mode == self.POSITION_CONTROL_MODE:
            # -1~1 -> 0~4095
            return ((actions + 1.0) * (self.POSITION_RANGE / 2)).astype(int)
        elif self.control_mode == self.VELOCITY_CONTROL_MODE:
            # -1~1 -> 0~1023 (속도는 양수만)
            return (np.abs(actions) * self.VELOCITY_RANGE).astype(int)
        elif self.control_mode == self.CURRENT_CONTROL_MODE:
            # -1~1 -> -1193~1193
            return (actions * self.CURRENT_RANGE).astype(int)
    
    def shutdown(self):
        """정리 및 종료"""
        self.stop_real_time_reading()
        
        # 모든 모터 토크 비활성화
        for motor_id in self.motor_ids:
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0)
        
        # 포트 닫기
        self.port_handler.closePort()
        print("Dynamixel 인터페이스 종료")


# 사용 예제
if __name__ == "__main__":
    # 4개 모터 사용 예제
    motor_ids = [1, 2, 3, 4]
    
    # 하드웨어 인터페이스 초기화
    hw_interface = DynamixelXL430Interface(
        motor_ids=motor_ids,
        device_name='/dev/ttyUSB0',  # Linux
        # device_name='COM3',        # Windows
        control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
    )
    
    try:
        # 실시간 읽기 시작
        hw_interface.start_real_time_reading()
        
        # 테스트: 모터들을 중간 위치로 이동
        test_positions = np.array([2048, 2048, 2048, 2048])  # 중간 위치
        hw_interface.set_position_commands(test_positions)
        
        # 5초간 상태 모니터링
        for _ in range(50):
            obs = hw_interface.get_observations()
            normalized_obs = hw_interface.normalize_observations(obs)
            
            print(f"Position: {obs['joint_pos']}")
            print(f"Normalized: {normalized_obs['joint_pos']}")
            time.sleep(0.1)
    
    finally:
        hw_interface.shutdown()