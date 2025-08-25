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
                if len(parts) >= 4:
                    return {
                        'timestamp': int(parts[0]),
                        'value': float(parts[1]),
                        'button': int(parts[2]),
                        'tare': int(parts[3])
                    }
        except:
            pass
        return None
    
    def run(self):
        print("이중 로드셀 데이터 수신 중... (Ctrl+C로 종료)")
        
        try:
            while True:
                data = self.read_data()
                if data:
                    print(f"처리된 값: {data['value']:8.3f}g | "
                          f"버튼: {'●' if data['button']==0 else '○'} | "
                          f"Tare: {'✓' if data['tare'] else '✗'}")
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n종료")
            self.ser.close()

if __name__ == "__main__":
    try:
        reader = SimpleLoadCell()
        reader.run()
    except Exception as e:
        print(f"연결 실패: {e}")