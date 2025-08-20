# Auto Balancing Case Sim2Real

Isaac Lab에서 학습한 RL Policy를 실제 Auto Balancing Case 하드웨어에서 실행하는 Sim2Real 시스템입니다.

## 시스템 구조

```
suitcase_sim2real/run/
├── run_policy.py                    # 메인 실행 스크립트
├── auto_balancing_case_bridge.py    # RL Policy와 하드웨어 간 브릿지
├── dynamixel_xl430_interface.py     # Dynamixel XL430 모터 인터페이스
├── hx711_interface.py               # HX711 Load Cell 인터페이스
├── config/
│   └── policy_config.yml            # 설정 파일
└── README.md                        # 이 파일
```

## 하드웨어 요구사항

### 액추에이터
- **Dynamixel XL430**: 밸런싱 힌지 조인트용 서보모터
- **USB2Dynamixel 또는 U2D2**: Dynamixel 통신 어댑터

### 센서
- **HX711 Load Cell Amplifier**: 5개 (바퀴 4개 + 손잡이 1개)
- **Load Cell**: 5개 (각 바퀴와 손잡이에 부착)
- **Raspberry Pi**: GPIO 핀을 통한 HX711 제어

### 연결
- Dynamixel XL430 ↔ USB2Dynamixel ↔ USB (컴퓨터)
- Load Cells ↔ HX711 ↔ GPIO (Raspberry Pi)

## 소프트웨어 요구사항

### Python 패키지
```bash
pip install torch numpy pyyaml
pip install dynamixel-sdk
pip install RPi.GPIO hx711  # Raspberry Pi에서만
```

### Isaac Lab 환경
학습에 사용된 Isaac Lab 환경과 동일한 observation/action 구조를 유지합니다:

**Observations:**
- `joint_pos`: 밸런싱 힌지 조인트 위치 (1D, 라디안)
- `joint_vel`: 밸런싱 힌지 조인트 속도 (1D, 라디안/초)
- `prev_action`: 이전 액션 (1D)
- `wheel_contact_forces`: 4개 바퀴의 접촉력 크기 (4D, 뉴턴, 순서: FR, RR, FL, RL)
- `handle_external_force`: 손잡이에 가해진 외부 힘 크기 (1D, 뉴턴)

**History Length**: 각 observation은 4 timestep의 history를 가짐 (총 8×4=32차원)

**Actions:**
- `hinge_pos`: 밸런싱 힌지 조인트 목표 위치 (1D, -0.5~0.5 라디안)

## 설정

### 1. 설정 파일 생성
```bash
cd suitcase_sim2real/run
python run_policy.py --create-config
```

### 2. 설정 파일 수정
`config/policy_config.yml` 파일을 실제 하드웨어에 맞게 수정:

```yaml
policy:
  model_path: "/path/to/your/rsl_rl_checkpoint.pt"  # 실제 체크포인트 경로

hardware:
  motor:
    id: 1                    # Dynamixel ID
    device: "/dev/ttyUSB0"   # 시리얼 포트
  
  load_cells:
    - name: "wheel_FR"      # IsaacLab USD 순서와 일치
      dout_pin: 5           # 실제 GPIO 핀 번호
      pd_sck_pin: 6
```

## 사용법

### 1. Load Cell 캘리브레이션 (처음 한 번만)
```bash
python run_policy.py --mode calibrate
```

지시에 따라 1kg 추를 사용하여 각 load cell을 캘리브레이션합니다.

### 2. 에피소드 실행 (권장)
```bash
python run_policy.py --mode episode
```

한 에피소드 (최대 8초, IsaacLab과 동일) 동안 RL policy를 실행합니다.

### 3. 연속 실행
```bash
python run_policy.py --mode continuous
```

Ctrl+C로 중단할 때까지 계속 실행됩니다.

### 4. 사용자 정의 설정 파일
```bash
python run_policy.py --config /path/to/custom_config.yml
```

## 안전 기능

### 자동 비상 정지
- 조인트 각도가 ±0.4 라디안(±23도) 초과 시 자동 정지
- 에피소드 최대 길이 도달 시 자동 종료
- Ctrl+C로 언제든 수동 중단 가능

### 안전한 종료
- 프로그램 종료 시 모터를 중앙 위치(0도)로 이동
- 모든 토크 비활성화 후 포트 정리

## 문제 해결

### 1. Dynamixel 연결 오류
```
RuntimeError: 포트 열기 실패
```
- USB 케이블 연결 확인
- 장치 경로 확인 (`ls /dev/ttyUSB*`)
- 사용자 권한 확인 (`sudo usermod -a -G dialout $USER`)

### 2. Load Cell 읽기 오류
```
Load cell 'wheel_FL' 읽기 오류
```
- GPIO 핀 연결 확인
- HX711 전원 공급 확인  
- 캘리브레이션 데이터 확인

### 3. Policy 로드 오류
```
Policy file not found
```
- 체크포인트 파일 경로 확인
- 파일 접근 권한 확인
- Isaac Lab RSL-RL 형식 확인

## 성능 튜닝

### 제어 주파수 조정
```yaml
policy:
  control_frequency: 50.0  # 기본값, 필요시 조정
```

### 정규화 파라미터 조정
```yaml
normalization:
  max_wheel_force: 50.0    # 실제 측정값에 맞게 조정
  max_handle_force: 20.0
  max_joint_angle: 0.5
```

## 개발자 정보

### 코드 구조
- `auto_balancing_case_bridge.py`: 메인 로직, policy 실행, observation/action 변환
- `dynamixel_xl430_interface.py`: Dynamixel 모터 저수준 제어
- `hx711_interface.py`: Load cell 데이터 읽기 및 캘리브레이션
- `run_policy.py`: CLI 인터페이스 및 설정 관리

### 확장 가능성
- 다른 모터 타입 지원 (interface 교체)
- 다른 센서 타입 지원 (force sensor 교체)
- 로깅 및 데이터 수집 기능 추가
- 실시간 시각화 기능 추가

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
