#!/usr/bin/env python3
"""
Dynamixel XL430 Hardware Interface for RL Policy Integration
XL430은 X-Series Protocol 2.0을 사용
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    print("Warning: dynamixel_sdk not installed. Install with: pip install dynamixel-sdk")
    # Mock classes for development without hardware
    class PortHandler:
        def __init__(self, port): pass
        def openPort(self): return True
        def setBaudRate(self, baudrate): return True
        def closePort(self): pass
    
    class PacketHandler:
        def __init__(self, protocol): pass
        def write1ByteTxRx(self, port, id, addr, val): return 0, 0
        def write4ByteTxRx(self, port, id, addr, val): return 0, 0
        def read4ByteTxRx(self, port, id, addr): return 0, 0, 0
        def getTxRxResult(self, result): return "Mock result"
        def getRxPacketError(self, error): return "Mock error"
    
    COMM_SUCCESS = 0

class DynamixelXL430Interface:
    """Auto Balancing Case용 XL430 모터 인터페이스 (단일 밸런싱 조인트)"""
    
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
    
    # Auto Balancing Case 특정 설정
    CENTER_POSITION = 2048  # 중앙 위치 (0도)
    MAX_ANGLE_RAD = 0.5     # 최대 각도 (라디안, ±0.5rad = ±28.6도)
    POSITION_PER_RAD = POSITION_RANGE / (2 * np.pi)  # 1 라디안당 위치값
    
    def __init__(self, motor_id: int = 1, device_name: str = '/dev/ttyUSB0', 
                 baudrate: int = 57600, control_mode: int = POSITION_CONTROL_MODE):
        """
        Auto Balancing Case용 단일 모터 인터페이스
        
        Args:
            motor_id: 밸런싱 모터 ID (기본값: 1)
            device_name: 시리얼 포트 (Linux: /dev/ttyUSB0, Windows: COM3)
            baudrate: 통신 속도
            control_mode: 제어 모드 (위치/속도/전류)
        """
        self.motor_id = motor_id
        self.control_mode = control_mode
        
        # SDK 초기화
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(2.0)  # Protocol 2.0
        
        # 상태 저장 변수들 (단일 모터)
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_current = 0.0
        self.goal_position = self.CENTER_POSITION  # 중앙 위치로 초기화
        
        # 실시간 읽기를 위한 변수들
        self.is_reading = False
        self.read_thread = None
        self.read_frequency = 100  # Hz
        
        self._initialize_connection()
        self._setup_motor()
    
    def _initialize_connection(self):
        """시리얼 연결 초기화"""
        if not self.port_handler.openPort():
            raise RuntimeError("포트 열기 실패")
        
        if not self.port_handler.setBaudRate(57600):
            raise RuntimeError("Baudrate 설정 실패")
        
        print("Dynamixel 연결 성공")
    
    def _setup_motor(self):
        """밸런싱 모터 초기 설정"""
        # 토크 비활성화
        self._write_1byte(self.motor_id, self.ADDR_TORQUE_ENABLE, 0)
        
        # 제어 모드 설정
        self._write_1byte(self.motor_id, self.ADDR_OPERATING_MODE, self.control_mode)
        
        # 중앙 위치로 이동
        self._write_4byte(self.motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION)
        
        # 토크 활성화
        self._write_1byte(self.motor_id, self.ADDR_TORQUE_ENABLE, 1)
        
        print(f"밸런싱 모터 {self.motor_id} 초기화 완료 (중앙 위치: {self.CENTER_POSITION})")
    
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
            
            # 밸런싱 모터의 현재 상태 읽기
            # Position 읽기 (4바이트, 0-4095)
            pos = self._read_4byte(self.motor_id, self.ADDR_PRESENT_POSITION)
            self.current_position = pos
            
            # Velocity 읽기 (4바이트, signed)
            vel = self._read_4byte(self.motor_id, self.ADDR_PRESENT_VELOCITY)
            # 부호 있는 32비트로 변환
            if vel > 2147483647:
                vel -= 4294967296
            self.current_velocity = vel
            
            # Current 읽기 (2바이트, signed)
            curr = self._read_4byte(self.motor_id, self.ADDR_PRESENT_CURRENT)
            if curr > 32767:
                curr -= 65536
            self.current_current = curr
            
            # 주기 유지
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def set_position_command(self, position: float):
        """위치 명령 전송 (RL Policy의 action을 모터 명령으로 변환)
        
        Args:
            position: 목표 위치 (0-4095 범위)
        """
        # Position 범위 확인 및 클리핑
        pos_cmd = int(np.clip(position, 0, self.POSITION_RANGE))
        self._write_4byte(self.motor_id, self.ADDR_GOAL_POSITION, pos_cmd)
        self.goal_position = pos_cmd
    
    def set_angle_command(self, angle_rad: float):
        """각도 명령 전송 (라디안 단위)
        
        Args:
            angle_rad: 목표 각도 (라디안, -0.5~0.5 범위)
        """
        # 각도를 위치값으로 변환
        angle_rad = np.clip(angle_rad, -self.MAX_ANGLE_RAD, self.MAX_ANGLE_RAD)
        position = self.CENTER_POSITION + int(angle_rad * self.POSITION_PER_RAD)
        self.set_position_command(position)
    
    def set_velocity_command(self, velocity: float):
        """속도 명령 전송"""
        if self.control_mode != self.VELOCITY_CONTROL_MODE:
            print("Warning: 속도 제어 모드가 아닙니다")
        
        # Velocity 범위 확인 및 클리핑
        vel_cmd = int(np.clip(velocity, 0, self.VELOCITY_RANGE))
        self._write_4byte(self.motor_id, self.ADDR_GOAL_VELOCITY, vel_cmd)
    
    def set_current_command(self, current: float):
        """전류 명령 전송"""
        if self.control_mode != self.CURRENT_CONTROL_MODE:
            print("Warning: 전류 제어 모드가 아닙니다")
        
        # Current 범위 확인 및 클리핑
        curr_cmd = int(np.clip(current, -self.CURRENT_RANGE, self.CURRENT_RANGE))
        if curr_cmd < 0:
            curr_cmd += 65536  # 2의 보수
        self._write_4byte(self.motor_id, self.ADDR_GOAL_CURRENT, curr_cmd)
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """RL Policy를 위한 observation 데이터 반환"""
        return {
            'joint_pos': np.array([self.current_position]),
            'joint_vel': np.array([self.current_velocity]), 
            'joint_effort': np.array([self.current_current]),
            'joint_pos_target': np.array([self.goal_position])
        }
    
    def get_joint_angle_rad(self) -> float:
        """현재 조인트 각도를 라디안으로 반환"""
        angle_rad = (self.current_position - self.CENTER_POSITION) / self.POSITION_PER_RAD
        return angle_rad
    
    def get_joint_velocity_rad_s(self) -> float:
        """현재 조인트 각속도를 라디안/초로 반환"""
        # XL430의 velocity 단위를 라디안/초로 변환 (대략적인 변환)
        # 실제 변환 계수는 모터 사양서를 참조하여 조정 필요
        velocity_rad_s = self.current_velocity * 0.01  # 임시 변환 계수
        return velocity_rad_s
    
    def normalize_observations(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Observation을 RL에서 사용하는 정규화된 범위로 변환"""
        normalized = {}
        
        # Position: 중앙 위치 기준으로 라디안 단위로 정규화
        angle_rad = (obs['joint_pos'] - self.CENTER_POSITION) / self.POSITION_PER_RAD
        normalized['joint_pos'] = angle_rad / self.MAX_ANGLE_RAD  # -1 to 1
        
        # Velocity: 라디안/초 단위로 정규화 (최대 각속도 기준)
        max_angular_vel = 6.0  # 라디안/초 (조정 가능)
        velocity_rad_s = obs['joint_vel'] * 0.01  # 임시 변환 계수
        normalized['joint_vel'] = np.clip(velocity_rad_s / max_angular_vel, -1.0, 1.0)
        
        # Current: -1193~1193 -> -1 to 1  
        normalized['joint_effort'] = obs['joint_effort'] / self.CURRENT_RANGE
        
        # Target position
        target_angle_rad = (obs['joint_pos_target'] - self.CENTER_POSITION) / self.POSITION_PER_RAD
        normalized['joint_pos_target'] = target_angle_rad / self.MAX_ANGLE_RAD
        
        return normalized
    
    def denormalize_actions(self, actions: np.ndarray) -> float:
        """RL Policy의 정규화된 action을 실제 모터 명령으로 변환
        
        Args:
            actions: 정규화된 action (-1~1, 라디안 기준)
            
        Returns:
            모터 위치 명령값 (0~4095)
        """
        if self.control_mode == self.POSITION_CONTROL_MODE:
            # 정규화된 action을 라디안으로 변환 후 모터 위치로 변환
            angle_rad = actions[0] * self.MAX_ANGLE_RAD  # -0.5~0.5 rad
            position = self.CENTER_POSITION + int(angle_rad * self.POSITION_PER_RAD)
            return np.clip(position, 0, self.POSITION_RANGE)
        elif self.control_mode == self.VELOCITY_CONTROL_MODE:
            # -1~1 -> 0~1023 (속도는 양수만)
            return int(np.abs(actions[0]) * self.VELOCITY_RANGE)
        elif self.control_mode == self.CURRENT_CONTROL_MODE:
            # -1~1 -> -1193~1193
            return int(actions[0] * self.CURRENT_RANGE)
    
    def shutdown(self):
        """정리 및 종료"""
        self.stop_real_time_reading()
        
        # 모터를 중앙 위치로 이동 후 토크 비활성화
        self._write_4byte(self.motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION)
        time.sleep(1.0)  # 이동 완료 대기
        self._write_1byte(self.motor_id, self.ADDR_TORQUE_ENABLE, 0)
        
        # 포트 닫기
        self.port_handler.closePort()
        print("Dynamixel 인터페이스 종료")


# 사용 예제
if __name__ == "__main__":
    # Auto Balancing Case용 단일 모터 예제
    motor_id = 1
    
    # 하드웨어 인터페이스 초기화
    hw_interface = DynamixelXL430Interface(
        motor_id=motor_id,
        device_name='/dev/ttyUSB0',  # Linux
        # device_name='COM3',        # Windows
        control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
    )
    
    try:
        # 실시간 읽기 시작
        hw_interface.start_real_time_reading()
        
        # 테스트: 모터를 다양한 각도로 이동
        test_angles = [0.0, 0.3, -0.3, 0.0]  # 라디안
        
        for angle in test_angles:
            print(f"목표 각도: {angle:.2f} rad ({angle*180/np.pi:.1f}도)")
            hw_interface.set_angle_command(angle)
            
            # 2초간 상태 모니터링
            for i in range(20):
                obs = hw_interface.get_observations()
                normalized_obs = hw_interface.normalize_observations(obs)
                current_angle = hw_interface.get_joint_angle_rad()
                
                print(f"  현재 각도: {current_angle:.3f} rad ({current_angle*180/np.pi:.1f}도)")
                time.sleep(0.1)
            
            time.sleep(1.0)
    
    finally:
        hw_interface.shutdown()