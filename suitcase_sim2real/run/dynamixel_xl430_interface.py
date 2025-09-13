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
    """Auto Balancing Case용 XL430 쿼드 모터 인터페이스 (4개 모터로 구성된 밸런싱 조인트)"""
    
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
    
    def __init__(self, motor_ids: list = [1, 2, 3, 4], 
                 device_names: list = ['/dev/ttyUSB0', '/dev/ttyUSB1'], 
                 baudrate: int = 57600, control_mode: int = POSITION_CONTROL_MODE):
        """
        Auto Balancing Case용 쿼드 모터 인터페이스 (4개 모터, 2개 포트)
        
        Args:
            motor_ids: 밸런싱 모터 ID 리스트 (기본값: [1, 2, 3, 4])
                      첫 2개는 +theta, 뒤 2개는 -theta 제어
            device_names: 시리얼 포트 리스트 (Linux: ['/dev/ttyUSB0', '/dev/ttyUSB1'], Windows: ['COM11', 'COM12'])
            baudrate: 통신 속도
            control_mode: 제어 모드 (위치/속도/전류)
        """
        self.motor_ids = motor_ids
        self.control_mode = control_mode
        
        # 포트별 모터 그룹 분리 (포트1: [1,2], 포트2: [3,4])
        self.port1_motors = motor_ids[:2]  # [1, 2]
        self.port2_motors = motor_ids[2:]  # [3, 4]
        
        self.port_lock = threading.Lock()
        # SDK 초기화 - 2개 포트
        self.port_handler1 = PortHandler(device_names[0])  # 포트1: 모터 [1,2]
        self.port_handler2 = PortHandler(device_names[1])  # 포트2: 모터 [3,4]
        self.packet_handler = PacketHandler(2.0)  # Protocol 2.0
        
        # 상태 저장 변수들 (쿼드 모터)
        self.current_positions = {motor_id: 0.0 for motor_id in motor_ids}
        self.current_velocities = {motor_id: 0.0 for motor_id in motor_ids}
        self.current_currents = {motor_id: 0.0 for motor_id in motor_ids}
        self.goal_positions = {motor_id: self.CENTER_POSITION for motor_id in motor_ids}
        
        # 실시간 읽기를 위한 변수들
        self.is_reading = False
        self.read_thread = None
        self.read_frequency = 50  # Hz
        
        self._initialize_connection()
        self._setup_motor()
    
    def _initialize_connection(self):
        """시리얼 연결 초기화 (2개 포트)"""
        # 포트1 초기화 (모터 [1,2])
        if not self.port_handler1.openPort():
            raise RuntimeError(f"포트1 열기 실패: {self.port_handler1}")
        
        if not self.port_handler1.setBaudRate(57600):
            raise RuntimeError("포트1 Baudrate 설정 실패")
        
        # 포트2 초기화 (모터 [3,4])
        if not self.port_handler2.openPort():
            raise RuntimeError(f"포트2 열기 실패: {self.port_handler2}")
        
        if not self.port_handler2.setBaudRate(57600):
            raise RuntimeError("포트2 Baudrate 설정 실패")
        
        print("Dynamixel 2포트 연결 성공")
    
    def _setup_motor(self):
        """쿼드 밸런싱 모터 초기 설정 (포트별로 설정)"""
        # 포트1 모터들 설정 (모터 [1,2])
        for motor_id in self.port1_motors:
            # 1. 먼저 토크 비활성화 (필수!)
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0, port_handler=self.port_handler1)
            time.sleep(0.01)  # 설정 안정화 대기
            
            # 2. 제어 모드 설정 (토크 비활성화 상태에서만 가능)
            self._write_1byte(motor_id, self.ADDR_OPERATING_MODE, self.control_mode, port_handler=self.port_handler1)
            time.sleep(0.01)
            
            # 3. 토크 활성화
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 1, port_handler=self.port_handler1)
            time.sleep(0.01)
            
            # 4. 중앙 위치로 이동 (토크 활성화 후)
            self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION, port_handler=self.port_handler1)
            
            print(f"밸런싱 모터 {motor_id} (포트1) 초기화 완료 (중앙 위치: {self.CENTER_POSITION})")
        
        # 포트2 모터들 설정 (모터 [3,4])
        for motor_id in self.port2_motors:
            # 1. 먼저 토크 비활성화 (필수!)
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0, port_handler=self.port_handler2)
            time.sleep(0.01)  # 설정 안정화 대기
            
            # 2. 제어 모드 설정 (토크 비활성화 상태에서만 가능)
            self._write_1byte(motor_id, self.ADDR_OPERATING_MODE, self.control_mode, port_handler=self.port_handler2)
            time.sleep(0.01)
            
            # 3. 토크 활성화
            self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 1, port_handler=self.port_handler2)
            time.sleep(0.01)
            
            # 4. 중앙 위치로 이동 (토크 활성화 후)
            self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION, port_handler=self.port_handler2)
            
            print(f"밸런싱 모터 {motor_id} (포트2) 초기화 완료 (중앙 위치: {self.CENTER_POSITION})")
    
    def _write_1byte(self, motor_id: int, address: int, value: int, port_handler=None):
        """1바이트 데이터 쓰기 (스레드 안전)"""
        if port_handler is None:
            # 모터 ID에 따라 적절한 포트 핸들러 선택
            port_handler = self.port_handler1 if motor_id in self.port1_motors else self.port_handler2
        
        with self.port_lock:  # 뮤텍스 사용
            result, error = self.packet_handler.write1ByteTxRx(
                port_handler, motor_id, address, value)
            if result != COMM_SUCCESS:
                print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
            if error != 0:
                print(f"Hardware error: {self.packet_handler.getRxPacketError(error)}")

    def _write_4byte(self, motor_id: int, address: int, value: int, port_handler=None):
        """4바이트 데이터 쓰기 (스레드 안전)"""
        if port_handler is None:
            # 모터 ID에 따라 적절한 포트 핸들러 선택
            port_handler = self.port_handler1 if motor_id in self.port1_motors else self.port_handler2
        
        with self.port_lock:  # 뮤텍스 사용
            result, error = self.packet_handler.write4ByteTxRx(
                port_handler, motor_id, address, value)
            if result != COMM_SUCCESS:
                print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
            if error != 0:
                print(f"Hardware error: {self.packet_handler.getRxPacketError(error)}")
    
    def _read_4byte(self, motor_id: int, address: int, port_handler=None) -> int:
        """4바이트 데이터 읽기 (스레드 안전)"""
        if port_handler is None:
            # 모터 ID에 따라 적절한 포트 핸들러 선택
            port_handler = self.port_handler1 if motor_id in self.port1_motors else self.port_handler2
        
        with self.port_lock:  # 뮤텍스 사용
            value, result, error = self.packet_handler.read4ByteTxRx(
                port_handler, motor_id, address)
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
            
            # 모든 밸런싱 모터의 현재 상태 읽기 (4개 모터)
            for motor_id in self.motor_ids:
                # Position 읽기 (4바이트, 0-4095)
                pos = self._read_4byte(motor_id, self.ADDR_PRESENT_POSITION)
                self.current_positions[motor_id] = pos
                
                # Velocity 읽기 (4바이트, signed)
                vel = self._read_4byte(motor_id, self.ADDR_PRESENT_VELOCITY)
                # 부호 있는 32비트로 변환
                if vel > 2147483647:
                    vel -= 4294967296
                self.current_velocities[motor_id] = vel
                
                # Current 읽기 (2바이트, signed)
                curr = self._read_4byte(motor_id, self.ADDR_PRESENT_CURRENT)
                if curr > 32767:
                    curr -= 65536
                self.current_currents[motor_id] = curr
            
            # 주기 유지
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def set_command(self, action: float):
        """통합 명령 전송 함수 - 4개 모터에 +@, -@ 분배
        
        Args:
            action: RL Policy의 action (라디안 단위, bridge에서 이미 clipping됨)
        """
        # 각도를 위치값으로 변환 (bridge에서 이미 clipping되었으므로 추가 clipping 불필요)
        angle_rad = action
        
        # 4개 모터에 +theta, +theta, -theta, -theta 분배
        for i, motor_id in enumerate(self.motor_ids):
            if i < 2:  # 첫 2개 모터: +theta
                position = int(self.CENTER_POSITION + angle_rad * self.POSITION_PER_RAD)
            else:  # 뒤 2개 모터: -theta
                position = int(self.CENTER_POSITION - angle_rad * self.POSITION_PER_RAD)
            
            # bridge에서 이미 action을 적절한 범위로 clipping했으므로 추가 clipping 불필요
            # 단, 안전을 위해 하드웨어 한계값만 체크
            if position < 0 or position > self.POSITION_RANGE:
                print(f"Warning: Position {position} out of range [0, {self.POSITION_RANGE}] for motor {motor_id}")
                position = max(0, min(position, self.POSITION_RANGE))
            
            self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, position)
            self.goal_positions[motor_id] = position

    def get_state(self) -> Dict[str, float]:
        """통합 상태 반환 함수 - 4개 모터의 평균값 계산 (-@ 모터도 +@로 간주)
        
        Returns:
            Dict: position (라디안), velocity (라디안/초) 상태값
        """
        # 4개 모터의 위치값을 +@로 통일하여 평균 계산
        positions = []
        velocities = []
        
        for i, motor_id in enumerate(self.motor_ids):
            pos = self.current_positions[motor_id]
            vel = self.current_velocities[motor_id]
            
            if i < 2:  # 첫 2개 모터: +theta (그대로 사용)
                positions.append(pos)
                velocities.append(vel)
            else:  # 뒤 2개 모터: -theta (부호 반전하여 +@로 변환)
                # -theta 모터의 위치를 +theta 기준으로 변환
                # position = CENTER - angle*POSITION_PER_RAD 이므로
                # angle = (CENTER - position) / POSITION_PER_RAD
                # 이를 +theta로 변환하면: CENTER + angle*POSITION_PER_RAD
                angle_from_center = (self.CENTER_POSITION - pos) / self.POSITION_PER_RAD
                converted_pos = self.CENTER_POSITION + angle_from_center * self.POSITION_PER_RAD
                positions.append(converted_pos)
                velocities.append(-vel)  # 속도도 부호 반전
        
        # 평균값 계산
        avg_position = np.mean(positions)
        avg_velocity = np.mean(velocities)
        
        # 라디안 단위로 변환
        angle_rad = (avg_position - self.CENTER_POSITION) / self.POSITION_PER_RAD
        velocity_rad_s = avg_velocity * 0.01  # 임시 변환 계수 (모터 사양서 참조 필요)
        
        return {
            'position': angle_rad,
            'velocity': velocity_rad_s
        }
    
    def shutdown(self):
        """정리 및 종료"""
        try:
            self.stop_real_time_reading()
            
            # 모든 모터를 중앙 위치로 이동 후 토크 비활성화 (4개 모터)
            for motor_id in self.motor_ids:
                self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION)
            time.sleep(1.0)
            
            for motor_id in self.motor_ids:
                self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0)
                
        except Exception as e:
            print(f"Shutdown error: {e}")
        finally:
            # 2개 포트 모두 강제 종료
            if hasattr(self, 'port_handler1') and self.port_handler1:
                self.port_handler1.closePort()
            if hasattr(self, 'port_handler2') and self.port_handler2:
                self.port_handler2.closePort()
            print("Dynamixel 쿼드 모터 인터페이스 종료 (2포트)")

# 사용 예제
if __name__ == "__main__":
    # Auto Balancing Case용 쿼드 모터 예제
    motor_ids = [1, 2, 3, 4]
    
    # 하드웨어 인터페이스 초기화 (2개 포트)
    hw_interface = DynamixelXL430Interface(
        motor_ids=motor_ids,
        device_names=['/dev/ttyUSB0', '/dev/ttyUSB1'],  # Linux: 2개 포트
        # device_names=['COM11', 'COM12'],              # Windows: 2개 포트
        control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
    )
    
    try:
        # 실시간 읽기 시작
        hw_interface.start_real_time_reading()
        
        # 테스트: 모터를 다양한 각도로 이동
        test_angles = [0.0, 0.3, -0.3, 0.0]  # 라디안
        
        for angle in test_angles:
            print(f"목표 각도: {angle:.2f} rad ({angle*180/np.pi:.1f}도)")
            hw_interface.set_command(angle)
            
            # 2초간 상태 모니터링
            for i in range(20):
                state = hw_interface.get_state()
                current_angle = state['position']
                current_velocity = state['velocity']
                
                print(f"  현재 각도: {current_angle:.3f} rad ({current_angle*180/np.pi:.1f}도), "
                      f"속도: {current_velocity:.3f} rad/s")
                time.sleep(0.1)
            
            time.sleep(1.0)
    
    finally:
        hw_interface.shutdown()