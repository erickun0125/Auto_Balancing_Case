#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import * # Uses Dynamixel SDK library

#********* DYNAMIXEL Model definition *********
MY_DXL = 'X_SERIES'       # X330, X430, X540, 2X430 (OpenRB-150과 호환)

# Control table address for X-Series
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_POSITION       = 132
DXL_MINIMUM_POSITION_VALUE  = 0         # 최소 위치값
DXL_MAXIMUM_POSITION_VALUE  = 4095      # 최대 위치값
BAUDRATE                    = 57600     # 기본 baudrate

# DYNAMIXEL Protocol Version (2.0 for X-Series)
PROTOCOL_VERSION            = 2.0

# Factory default ID of all DYNAMIXEL is 1
DXL_ID                      = 2

# COM11 포트 설정 (Windows의 경우)
DEVICENAME                  = 'COM11'   # Windows: "COM11"

TORQUE_ENABLE               = 1     # 토크 활성화
TORQUE_DISABLE              = 0     # 토크 비활성화
DXL_MOVING_STATUS_THRESHOLD = 20    # 움직임 완료 판단 기준

# 각도를 모터 위치값으로 변환하는 함수
def angle_to_position(angle_deg):
    """
    각도(도)를 모터 위치값으로 변환
    X-Series: 0~4095 = 0~360도
    중심점을 2048로 설정하여 -180~+180도 범위로 사용
    """
    # -180도 ~ +180도 범위를 0~4095로 매핑
    position = int(2048 + (angle_deg * 4095 / 360))
    # 범위 제한
    position = max(0, min(4095, position))
    return position

# 위치값을 각도로 변환하는 함수  
def position_to_angle(position):
    """
    모터 위치값을 각도(도)로 변환
    """
    angle = (position - 2048) * 360 / 4095
    return angle

# 목표 각도 시퀀스 (도 단위)
angle_sequence = [0, 90, 0, -90, 0]

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("성공적으로 포트를 열었습니다")
else:
    print("포트 열기 실패")
    print("아무 키나 눌러서 종료...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("성공적으로 baudrate를 변경했습니다")
else:
    print("Baudrate 변경 실패")
    print("아무 키나 눌러서 종료...")
    getch()
    quit()

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel이 성공적으로 연결되었습니다")

print("\n모터 제어를 시작합니다...")
print("시퀀스: 0° → 90° → 0° → -90° → 0° (각 1초 간격)")
print("ESC 키를 눌러서 중단할 수 있습니다\n")

try:
    # 무한 반복 (ESC로 중단 가능)
    while True:
        for i, target_angle in enumerate(angle_sequence):
            # 키보드 입력 확인 (Windows에서만 작동)
            if os.name == 'nt' and msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # ESC key
                    print("\n사용자가 중단했습니다.")
                    raise KeyboardInterrupt
            
            # 각도를 모터 위치값으로 변환
            goal_position = angle_to_position(target_angle)
            
            print(f"단계 {i+1}: {target_angle}도로 이동 (위치값: {goal_position})")
            
            # 목표 위치 전송
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_position
            )
            
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            
            # 모터가 목표 위치에 도달할 때까지 대기
            start_time = time.time()
            while True:
                # 현재 위치 읽기
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                    portHandler, DXL_ID, ADDR_PRESENT_POSITION
                )
                
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler.getRxPacketError(dxl_error))
                else:
                    current_angle = position_to_angle(dxl_present_position)
                    print(f"  현재 위치: {current_angle:.1f}도 (목표: {target_angle}도)", end='\r')
                
                # 목표 위치에 도달했는지 확인
                if abs(goal_position - dxl_present_position) <= DXL_MOVING_STATUS_THRESHOLD:
                    print(f"  목표 위치 도달! 현재: {current_angle:.1f}도")
                    break
                
                # 타임아웃 방지 (5초)
                if time.time() - start_time > 5.0:
                    print(f"  타임아웃: 목표 위치에 도달하지 못했습니다")
                    break
                
                time.sleep(0.01)  # 10ms 대기
            
            # 1초 대기
            time.sleep(1.0)
        
        print("\n한 사이클 완료. 다시 시작합니다...\n")

except KeyboardInterrupt:
    print("\n프로그램을 중단합니다...")

# Disable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("토크가 비활성화되었습니다")

# Close port
portHandler.closePort()
print("포트가 닫혔습니다")
