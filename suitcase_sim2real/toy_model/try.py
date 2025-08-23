#!/usr/bin/env python3
"""
로드셀 센서 데이터 실시간 수신 및 표시
Arduino와 시리얼 통신으로 센서 데이터를 받아서 처리
"""

import serial
import serial.tools.list_ports
import threading
import time
import queue
import sys
from datetime import datetime
import numpy as np

class LoadCellInterface:
    def __init__(self, port=None, baudrate=115200):
        self.port = port or self._find_arduino_port()
        self.baudrate = baudrate
        self.serial_conn = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data = None
        self.running = False
        self.thread = None
        
        # 통계용 변수
        self.packet_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def _find_arduino_port(self):
        """Arduino 포트 자동 탐지"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Arduino 관련 키워드 검색
            if any(keyword in port.description.lower() for keyword in 
                   ['arduino', 'ch340', 'cp210', 'ftdi', 'usb']):
                print(f"Arduino 포트 감지: {port.device} - {port.description}")
                return port.device
        
        # 기본값들 시도
        default_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', 'COM3', 'COM4', 'COM7']
        for port in default_ports:
            try:
                test_serial = serial.Serial(port, 115200, timeout=1)
                test_serial.close()
                print(f"기본 포트 사용: {port}")
                return port
            except:
                continue
        
        raise Exception("Arduino 포트를 찾을 수 없습니다.")
    
    def connect(self):
        """Arduino와 연결"""
        try:
            self.serial_conn = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)  # Arduino 초기화 대기
            
            # 초기화 확인
            self._wait_for_ready()
            
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Arduino 연결 성공: {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Arduino 연결 실패: {e}")
            return False
    
    def _wait_for_ready(self):
        """Arduino READY 신호 대기"""
        print("Arduino 초기화 대기 중...")
        timeout = time.time() + 500  # 10초 타임아웃
        
        while time.time() < timeout:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino: {line}")
                
                if line == "READY":
                    print("✅ Arduino 초기화 완료")
                    return
                elif line.startswith("ERROR"):
                    raise Exception(f"Arduino 에러: {line}")
        
        raise Exception("Arduino 초기화 타임아웃")
    
    def _read_loop(self):
        """백그라운드에서 데이터 수신"""
        buffer = ""
        
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    # 데이터 읽기
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # 완전한 라인들 처리
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            self._process_line(line)
                
                time.sleep(0.001)  # 1ms 대기
                
            except Exception as e:
                self.error_count += 1
                if self.error_count % 100 == 0:  # 100번마다 출력
                    print(f"⚠️  수신 에러 ({self.error_count}회): {e}")
                time.sleep(0.01)
    
    def _process_line(self, line):
        """한 줄의 데이터 처리"""
        try:
            # 상태 메시지 처리
            if line in ['TARE_STARTED', 'TARE_COMPLETE'] or line.startswith('STATUS:'):
                print(f"📟 Arduino: {line}")
                return
            
            # CSV 데이터 파싱: timestamp,load_cell,button,tare_status
            parts = line.split(',')
            if len(parts) == 4:
                data = {
                    'timestamp': int(parts[0]),
                    'load_cell': float(parts[1]),
                    'button': int(parts[2]),
                    'tare_status': int(parts[3]),
                    'receive_time': time.time()
                }
                
                # 최신 데이터 업데이트
                self.latest_data = data
                
                # 큐에 추가 (큐가 가득 차면 오래된 데이터 제거)
                if not self.data_queue.full():
                    self.data_queue.put(data)
                else:
                    try:
                        self.data_queue.get_nowait()  # 오래된 데이터 제거
                        self.data_queue.put(data)     # 새 데이터 추가
                    except queue.Empty:
                        pass
                
                self.packet_count += 1
                
        except ValueError as e:
            self.error_count += 1
    
    def get_latest_data(self):
        """최신 센서 데이터 반환"""
        return self.latest_data
    
    def get_data_stream(self, count=100):
        """최근 N개의 데이터 반환"""
        data_list = []
        temp_queue = queue.Queue()
        
        # 큐에서 데이터 추출
        while not self.data_queue.empty() and len(data_list) < count:
            try:
                data = self.data_queue.get_nowait()
                data_list.append(data)
                temp_queue.put(data)
            except queue.Empty:
                break
        
        # 데이터를 다시 큐에 넣기
        while not temp_queue.empty():
            self.data_queue.put(temp_queue.get())
        
        return data_list[-count:] if data_list else []
    
    def send_command(self, command):
        """Arduino에 명령 전송"""
        if self.serial_conn:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                return True
            except Exception as e:
                print(f"❌ 명령 전송 실패: {e}")
                return False
        return False
    
    def tare(self):
        """로드셀 영점 조정"""
        return self.send_command("TARE")
    
    def get_status(self):
        """Arduino 상태 요청"""
        return self.send_command("STATUS")
    
    def set_calibration(self, cal_factor):
        """Calibration factor 설정"""
        return self.send_command(f"CAL={cal_factor}")
    
    def set_samples(self, samples):
        """샘플링 수 설정"""
        return self.send_command(f"SAMPLES={samples}")
    
    def get_statistics(self):
        """통신 통계 반환"""
        elapsed = time.time() - self.start_time
        return {
            'packets': self.packet_count,
            'errors': self.error_count,
            'rate': self.packet_count / elapsed if elapsed > 0 else 0,
            'uptime': elapsed
        }
    
    def close(self):
        """연결 종료"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.serial_conn:
            self.serial_conn.close()
        print("🔌 Arduino 연결 종료")

def display_sensor_data():
    """센서 데이터 실시간 표시"""
    load_cell = LoadCellInterface()
    
    if not load_cell.connect():
        return
    
    print("\n" + "="*60)
    print("🔬 로드셀 센서 데이터 모니터링")
    print("="*60)
    print("명령어:")
    print("  t: Tare (영점조정)")
    print("  s: 상태 확인") 
    print("  q: 종료")
    print("  c: 통계 보기")
    print("="*60)
    
    try:
        last_display = 0
        
        while True:
            current_time = time.time()
            
            # 0.1초마다 화면 업데이트
            if current_time - last_display >= 0.1:
                data = load_cell.get_latest_data()
                
                if data:
                    # 시간 계산
                    arduino_time = data['timestamp'] / 1000.0  # ms → s
                    delay = (data['receive_time'] - time.time()) * 1000  # 지연시간 (ms)
                    
                    # 데이터 표시
                    print(f"\r🏋️  로드셀: {data['load_cell']:8.3f}g | "
                          f"버튼: {'🔴' if data['button'] else '⚪'} | "
                          f"Tare: {'✅' if data['tare_status'] else '❌'} | "
                          f"지연: {abs(delay):4.1f}ms", end="", flush=True)
                
                last_display = current_time
            
            # 키보드 입력 처리 (논블로킹)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip().lower()
                
                if key == 'q':
                    break
                elif key == 't':
                    print("\n🔄 Tare 시작...")
                    load_cell.tare()
                elif key == 's':
                    print("\n📊 상태 요청...")
                    load_cell.get_status()
                elif key == 'c':
                    stats = load_cell.get_statistics()
                    print(f"\n📈 통계: {stats['packets']}패킷, "
                          f"{stats['rate']:.1f}Hz, 에러 {stats['errors']}회")
            
            time.sleep(0.01)  # 10ms 대기
            
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        load_cell.close()

def simple_data_logger():
    """간단한 데이터 로거"""
    load_cell = LoadCellInterface()
    
    if not load_cell.connect():
        return
    
    print("📝 데이터 로깅 시작 (Ctrl+C로 중단)")
    
    try:
        while True:
            data = load_cell.get_latest_data()
            if data:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{timestamp} | 로드셀: {data['load_cell']:8.3f}g | "
                      f"버튼: {data['button']} | Tare: {data['tare_status']}")
            
            time.sleep(0.1)  # 100ms 간격
            
    except KeyboardInterrupt:
        print("\n📊 로깅 종료")
        stats = load_cell.get_statistics()
        print(f"총 {stats['packets']}개 패킷 수신, 평균 {stats['rate']:.1f}Hz")
    finally:
        load_cell.close()

if __name__ == "__main__":
    # select 모듈 import (키보드 입력용)
    try:
        import select
        display_sensor_data()
    except ImportError:
        # Windows에서는 select가 소켓에만 작동
        print("⚠️  Windows에서는 simple logger 모드로 실행")
        simple_data_logger()