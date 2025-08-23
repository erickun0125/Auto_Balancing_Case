# Auto Balancing Case Sim2Real

Isaac Lab에서 학습한 RL policy를 실제 Auto Balancing Case 하드웨어에서 실행하는 sim2real 프로젝트입니다.

## 하드웨어 구성

### 1. Actuator: Dynamixel XL430 듀얼 모터
- **모터 개수**: 2개 (ID: 1, 2)
- **제어 방식**: +theta, -theta 반대 방향 제어
- **연결**: 하나의 축으로 물리적 연결
- **통신**: RS485 (USB2Dynamixel 또는 U2D2 사용)
- **제어 모드**: Position Control Mode

### 2. Sensor: HX711 Load Cell (Arduino 기반)
- **Load Cell 개수**: 5개 (바퀴 4개 + 손잡이 1개)
- **센서 위치**: 
  - wheel_FR (Front Right)
  - wheel_RR (Rear Right) 
  - wheel_FL (Front Left)
  - wheel_RL (Rear Left)
  - handle (손잡이)
- **통신**: Arduino를 통한 시리얼 통신 (JSON 형태)

## 파일 구조

```
suitcase_sim2real/
├── run/
│   ├── auto_balancing_case_bridge.py      # 메인 브릿지 클래스
│   ├── dynamixel_xl430_interface.py       # 듀얼 모터 인터페이스
│   ├── hx711_interface.py                 # Arduino 기반 Load Cell 인터페이스
│   ├── run_policy.py                      # 정책 실행 메인 스크립트
│   └── config/
│       └── policy_config.yml              # 설정 파일
├── arduino/
│   └── hx711_load_cells/
│       └── hx711_load_cells.ino           # Arduino 코드
└── README.md
```

## 설치 및 설정

### 1. Python 의존성 설치
```bash
pip install pyserial dynamixel-sdk numpy torch pyyaml
```

### 2. Arduino 설정
1. Arduino IDE에서 HX711 라이브러리 설치
2. `arduino/hx711_load_cells/hx711_load_cells.ino` 업로드
3. 하드웨어 연결:
   - wheel_FR: DOUT=2, SCK=3
   - wheel_RR: DOUT=4, SCK=5
   - wheel_FL: DOUT=6, SCK=7
   - wheel_RL: DOUT=8, SCK=9
   - handle: DOUT=10, SCK=11

### 3. 설정 파일 생성
```bash
cd suitcase_sim2real/run
python run_policy.py --create-config
```

### 4. 설정 파일 수정
`config/policy_config.yml` 파일을 실제 하드웨어에 맞게 수정:
```yaml
policy:
  model_path: '/path/to/your/rsl_rl_checkpoint.pt'
  control_frequency: 50.0
  device: 'cpu'

hardware:
  motor:
    ids: [1, 2]  # 듀얼 모터 ID
    device: '/dev/ttyUSB0'  # Linux: /dev/ttyUSB0, Windows: COM3
    baudrate: 57600
  arduino:
    port: '/dev/ttyACM0'  # Linux: /dev/ttyACM0, Windows: COM4
    baudrate: 115200

observation:
  history_length: 4

normalization:
  max_wheel_force: 50.0
  max_handle_force: 20.0
  max_joint_angle: 0.5
  max_joint_velocity: 6.0

safety:
  max_episode_steps: 400
  emergency_angle_limit: 0.4
```

## 사용법

### 1. Load Cell 캘리브레이션
```bash
python run_policy.py --mode calibrate
```

### 2. 한 에피소드 실행
```bash
python run_policy.py --mode episode
```

### 3. 연속 실행
```bash
python run_policy.py --mode continuous
```

## 주요 변경사항

### 1. Dynamixel 모터 인터페이스 (듀얼 모터 지원)
- **이전**: 단일 모터 (ID: 1)
- **현재**: 듀얼 모터 (ID: [1, 2])
- **제어 방식**: 
  - 모터 1: +theta 방향
  - 모터 2: -theta 방향 (반대 방향)
- **관측값**: 두 모터의 평균값 사용

### 2. HX711 Load Cell 인터페이스 (Arduino 기반)
- **이전**: Raspberry Pi GPIO 직접 제어
- **현재**: Arduino를 통한 시리얼 통신
- **데이터 형태**: JSON 형태로 전송
- **통신 속도**: 115200 baud, 50Hz 업데이트

### 3. 설정 파일 구조 변경
- 듀얼 모터 설정 추가
- Arduino 포트 설정 추가
- GPIO 핀 설정 제거

## Arduino 통신 프로토콜

### 데이터 수신 (50Hz)
```json
{"wheel_FR":123.45,"wheel_RR":67.89,"wheel_FL":234.56,"wheel_RL":78.90,"handle":12.34}
```

### 명령 전송
- `TARE_ALL`: 모든 Load Cell 영점 조정
- `SET_CAL_FR <value>`: FR 바퀴 캘리브레이션 계수 설정
- `SET_CAL_RR <value>`: RR 바퀴 캘리브레이션 계수 설정
- `SET_CAL_FL <value>`: FL 바퀴 캘리브레이션 계수 설정
- `SET_CAL_RL <value>`: RL 바퀴 캘리브레이션 계수 설정
- `SET_CAL_HANDLE <value>`: 손잡이 캘리브레이션 계수 설정
- `GET_RAW`: Raw 값 확인 (디버깅용)

## 문제 해결

### 1. 모터 연결 문제
- Dynamixel 모터 ID 확인
- 시리얼 포트 권한 확인: `sudo chmod 666 /dev/ttyUSB0`
- 통신 속도 확인 (57600 baud)

### 2. Arduino 연결 문제
- Arduino 포트 확인: `ls /dev/ttyACM*`
- 시리얼 포트 권한 확인: `sudo chmod 666 /dev/ttyACM0`
- Arduino 코드 업로드 확인

### 3. Load Cell 캘리브레이션
- 영점 조정 후 알려진 무게로 캘리브레이션
- 캘리브레이션 데이터는 `load_cell_calibration.npz`에 저장
- 각 Load Cell별로 개별 캘리브레이션 가능

## 안전 기능

1. **비상 정지**: 조인트 각도가 제한값 초과 시 자동 정지
2. **에피소드 제한**: 최대 스텝 수 제한
3. **모터 안전**: 종료 시 중앙 위치로 이동 후 토크 비활성화
4. **통신 오류 처리**: 하드웨어 통신 오류 시 안전 모드 전환
