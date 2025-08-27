#!/usr/bin/env python3
import serial
import time

class SimpleLoadCell:
    def __init__(self, port='COM7'):
        self.ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        
        # READY 대기
        while True:
            line = self.ser.readline().decode().strip()
            print(f"Arduino: {line}")
            if line == "READY":
                print("Arduino 준비 완료")
                break
            elif "ERROR" in line:
                print(f"에러: {line}")
                return
    
    def read_data(self):
        try:
            line = self.ser.readline().decode().strip()
            if line and ',' in line:
                parts = line.split(',')
                
                # Arduino에서 보내는 형식: diff1,diff2,button,tare (4개 값)
                if len(parts) >= 4:
                    return {
                        'diff1': float(parts[0]),      # 로드셀1 차이값 (val1 - avg)
                        'diff2': float(parts[1]),      # 로드셀2 차이값 (val2 - avg)
                        'button': int(parts[2]),       # 버튼 상태 (0=눌림, 1=안눌림)
                        'tare': int(parts[3])          # Tare 상태 (1=진행중, 0=완료)
                    }
            elif line and not any(word in line for word in ['TARE_', 'STATUS:', 'ERROR']):
                # 시스템 메시지가 아닌 알 수 없는 형식
                print(f"알 수 없는 데이터: '{line}'")
                
        except Exception as e:
            print(f"파싱 에러: {e}, 라인: {line}")
        return None
    
    def run(self):
        print("로드셀 데이터 수신 중... (Ctrl+C로 종료)")
        print("Arduino에서 diff1, diff2, button, tare 값을 받습니다.")
        
        try:
            while True:
                data = self.read_data()
                if data:
                    print(f"차이1: {data['diff1']:>7.3f}g | "
                          f"차이2: {data['diff2']:>7.3f}g | "
                          f"버튼: {'●' if data['button']==0 else '○'} | "
                          f"Tare: {'진행중' if data['tare'] else '완료'}")
                else:
                    # 시스템 메시지 처리
                    line = self.ser.readline().decode().strip()
                    if line:
                        if "TARE_" in line or "STATUS:" in line:
                            print(f"시스템: {line}")
                        elif line and line != "":
                            print(f"기타: '{line}'")
                            
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n종료")
            self.ser.close()
    
    def send_command(self, cmd):
        """Arduino에 명령 전송"""
        self.ser.write(f"{cmd}\n".encode())
        self.ser.flush()
        print(f"명령 전송: {cmd}")

if __name__ == "__main__":
    try:
        reader = SimpleLoadCell()
        
        # 명령 테스트 (선택사항)
        print("\n명령 테스트:")
        reader.send_command("STATUS")
        time.sleep(0.5)
        
        # Tare 명령 테스트
        print("Tare 명령을 보내시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            reader.send_command("TARE")
            time.sleep(1)
        
        print("\n데이터 읽기 시작:")
        reader.run()
        
    except Exception as e:
        print(f"연결 실패: {e}")
        print("COM 포트 번호를 확인하세요.")