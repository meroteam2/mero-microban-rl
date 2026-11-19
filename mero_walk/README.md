# mero_walk — MERO Microban 실물 걷기 런타임

Rhoban Microban의 걷기 정책(`walk.onnx`)을 우리 로봇(Orange Pi Zero 2W + Pollen Robot HAT, XL330 19개 @ 115200)에서 돌리기 위한 도구 모음입니다.

| 도구 | 어디서 | 하는 일 |
|---|---|---|
| `python -m mero check` | 로봇 | 19개 모터 설정 점검(Operating Mode, Drive 역방향, Homing, Shutdown, 위치 제한, 펌웨어). `--fix`로 권장값 기록 |
| `python -m mero calib` | 로봇 | **원점·회전방향·IMU 장착 방향** 확인과 수정 → `calibration.json` |
| `viewer/mero_viewer.py` | 노트북 | **Gazebo**(또는 MuJoCo)에 로봇 디지털 트윈 표시. 실제 로봇과 똑같이 움직이는지 눈으로 비교 |
| `python -m mero run` | 로봇 | **SSH 키보드 조종**: 토크 ON/OFF, 걷기 시작/정지, 방향키 속도 |
| `python -m mero battery` | 로봇 | 배터리 전압. `--watch`는 백그라운드 감시(서비스) |
| `python -m mero bench` | 로봇 | 115200 버스에서 실제로 낼 수 있는 제어 주기 측정 |
| `python -m mero imu` | 로봇 | IMU 값(몸통 기준 중력·자이로) 보기 |

> 포복·일어서기는 이번 범위에서 뺐습니다. 걷기만 있습니다.

---

## 0. 로봇에 올리기

노트북(PowerShell 또는 WSL)에서:

```bash
scp -r mero_walk orangepi@<로봇IP>:~/
ssh orangepi@<로봇IP>
cd ~/mero_walk && bash install_robot.sh
```

- 파이썬은 골든 카드에 이미 있는 `~/microban/.venv`(Python 3.12, numpy·onnxruntime·pyserial·smbus2)를 그대로 씁니다.
- `install_robot.sh`가 `dialout`/`i2c` 그룹을 추가하면 **SSH를 다시 접속**해야 적용됩니다(그 전엔 `sudo`가 필요).
- 아래 명령은 전부 `cd ~/mero_walk` 후 `~/microban/.venv/bin/python -m mero ...`로 실행합니다. 길면 `alias mero='~/microban/.venv/bin/python -m mero'`를 `~/.bashrc`에 넣으세요.

**모터 프로그램은 한 번에 하나만** 돌아야 합니다. 이 도구들은 `/tmp/mero-dxl.lock`으로 서로 막아 주지만, `setup_motor.py` 같은 다른 스크립트는 못 막으니 같이 돌리지 마세요.

## 1. 점검

```bash
mero check      # 빨간 줄이 없어야 함. 있으면 내용 확인 후 mero check --fix
mero battery    # 모터별 전압과 배터리 상태
mero imu        # 세워 두면 중력(몸통) ≈ (0, 0, -1)
mero bench      # 권장 control.hz를 알려 줌
```

## 2. 캘리브레이션 (Gazebo로 확인)

### 노트북 준비 (한 번)

WSL2 Ubuntu(22.04/24.04)에서:

```bash
bash mero_walk/viewer/setup_laptop_wsl.sh      # Gazebo Harmonic + gz 파이썬 바인딩 + mujoco
```

### 실행

1. **로봇(SSH 터미널)**: `mero calib` — 모든 토크가 꺼진 상태로 시작합니다.
2. **노트북(WSL 터미널)**: `mero_walk/viewer/.venv/bin/python mero_walk/viewer/mero_viewer.py --robot <로봇IP>`
   - Gazebo 창이 뜨고 로봇 모델이 실제 로봇을 따라 움직입니다(몸통 기울기는 IMU 값).
   - 화면이 안 나오면 `--render-engine ogre`, 그래도 안 되면 `--viewer mujoco`(Windows 파이썬에서도 `pip install mujoco numpy` 후 실행 가능).
   - 로봇 없이 뷰어만 확인: `--demo`

### 순서

| 단계 | 키 | 확인할 것 |
|---|---|---|
| ① 관찰 | (기본) | 관절을 **손으로** 하나씩 움직여 보고, Gazebo 모델의 같은 관절이 **같은 방향**으로 움직이는지 봅니다. 반대면 `↑↓`로 그 관절을 고르고 `f`(부호 반전) |
| ② 조그 | `j`, `←→` | 고른 관절 **하나만** 토크를 켜고 5°씩 움직입니다. 시안색 고스트(목표)와 실제 로봇이 같은 쪽으로 가야 합니다. 다시 `j`로 끔 |
| ③ 원점 | `z` → `A` | 로봇을 **기준 자세**(다리 쭉 펴고, 팔은 몸통 옆에 곧게 내리고, 머리 정면)로 두면 고스트와 겹쳐야 합니다. '차이' 열이 작으면(±5°) 정상. 20° 넘게 차이 나면 **혼 조립 방향**을 먼저 의심하세요. 맞추려면 `A`(전체) 또는 `Z`(선택 관절). 둘 다 원점을 **90° 단위**로 맞추고 남는 작은 오차(혼 톱니 오차 등)는 남깁니다. 그 오차까지 없애려면 그 관절만 기준 자세로 정확히 두고 `x`(스냅 없이 정확히) |
| ④ IMU | `m` | 안내대로 ① 똑바로 세우기 → ② 엎드려 눕히기 → ③ 왼쪽 옆으로 눕히기에서 ENTER. 장착 방향이 기대값과 다르면 새 값을 제안합니다(y로 적용) |
| ⑤ 자이로 | `g` | 3초간 가만히 두고 바이어스 측정 |
| ⑤-1 서기 보정 | `t`, `↑↓`, `←→` | `run`과 같이 토크를 켜고 2초 동안 서는 자세로 갑니다(로봇을 잡고). 모델(시안 고스트 = 서는 자세)과 비교해 틀어진 관절을 `↑↓`로 고르고 `←→`로 **1°씩** 원점을 보정합니다(→ = 시뮬레이션 + 방향). '차이' 열은 서는 자세와의 오차. `t` 토크 OFF, `k` 비상 정지 |
| ⑥ 저장 | `s` | `calibration.json` 저장(이전 파일은 `.bak`) |

화면 아래쪽 "기대:" 줄이 IMU 부호 기준입니다: 앞으로 숙이면 중력 x+, 왼쪽으로 기울이면 y+, 위에서 봐서 반시계로 돌리면 gyro z+.

> 각도 정의: `시뮬레이션 각도 = 부호 × ((raw − 2048) × 2π/4096 − 원점)`. 기본 부호는 업스트림 `MOTOR_SIGN`, 원점 0. 모터 EEPROM(Homing Offset·Drive Mode)은 원칙적으로 건드리지 않고 파일로만 보정합니다.
>
> **예외: 기준 자세가 모터 경계(raw 0/4095)에 걸린 관절.** 혼이 약 180° 돌아가 조립되면 기준 자세가 raw 0 근처입니다. 위치 모드(3)에서는 토크를 켜는 순간 모터가 위치를 한 바퀴(0~4095) 안으로 다시 계산해서, 직전에 계산한 목표가 한 바퀴 떨어져 **관절이 360° 돕니다**. 혼을 다시 끼우는 게 가장 깔끔하고, 못 하면 그 관절을 `config.json`의 `serial.multiturn_joints`에 넣으세요. 그 모터는 확장 위치 모드(4)로 돌고(`check --fix`가 바꿔 줌), 런타임이 몇 바퀴째인지 계산해서 읽기·쓰기를 맞춥니다(토크가 켜져 있는 동안은 고정). 바꾼 뒤 `calib`에서 원점을 다시 잡으세요. Homing Offset은 이 문제를 못 고칩니다(XL330 위치 모드는 ±1024까지만 적용되고, 실제 엔코더 위치는 그대로 경계에 남음). 우리 로봇: `["right_shoulder_pitch", "right_shoulder_roll"]`.

## 3. 걷기

```bash
mero run
```

| 키 | 동작 |
|---|---|
| `t` | 토크 ON → 2초 동안 서는 자세로 이동 / 서 있을 때 **두 번** 누르면 토크 OFF |
| `SPACE` (`w`) | 걷기 시작 / 정지 (정지하면 1초 제자리걸음 후 서는 자세) |
| `↑ ↓` | 앞/뒤 속도 ±0.05 m/s (기본 최대 0.30, 뒤 0.20) |
| `← →` | 회전 ±0.2 rad/s |
| `a d` | 옆걸음 |
| `s` | 속도 0 |
| `k` / `ESC` | **비상 정지**(즉시 모든 토크 OFF) |
| `q` | 종료(서 있으면 두 번) |

처음 걸릴 때는 **로봇 위를 손으로 잡을 준비**를 하고, 제자리걸음(속도 0) → `↑` 한두 번(0.05~0.10 m/s) 순서로 올리세요.

**걷기 로그**: `SPACE`로 걷기를 시작하면 `logs/walk_YYYYmmdd_HHMMSS.csv`에 제어 틱마다 한 줄씩 기록하고, 멈춰서 서 있는 상태가 되거나 넘어지거나 토크가 꺼지면 닫습니다(상태줄에 경로 표시). 열: 시간·상태, 명령(cmd_des/cmd), IMU(acc, gyro_imu, gyro_body, grav, quat, 나이), 19개 관절 q/dq, 정책 관측 `obs_*`와 원시 행동 `act_*`, 모터 목표 `tgt_*`, 전압, 읽기 오류, 그 틱의 메시지(`event`). 같은 이름의 `.json`에 정책·게인·보정값이 들어 있습니다. 노트북으로 가져오기: `scp orangepi@<로봇IP>:mero_walk/logs/walk_* .` 끄려면 `config.json`에 `"log": {"walk": false}`.

자동 보호:
- **넘어짐**(몸통 60° 이상 기울어짐) → 토크 OFF. 세운 뒤 `t`.
- **과전류 추정**(19개 합 15 A 이상) → 토크 OFF. (업스트림 방식, 추가 통신 없이 위치 오차로 추정)
- **SSH 끊김** → 걷는 중이면 멈춰 선 뒤 토크 OFF하고 종료.
- **모터 통신 오류 3회 연속** → 토크 OFF.
- 관절이 범위 밖이면 토크를 켜지 않습니다(켜는 순간 튀는 것 방지).
- 노트북 뷰어를 켜 두면 걷는 동안에도 트윈이 보입니다(`--no-stream`으로 끔).

## 4. 배터리 보호 (2S Li-ion)

> **모터를 5V 레귤레이터로 공급하면(MERO 현재 구성) 아래 4-1을 먼저 보세요.** 모터가 재는 전압으로는 배터리를 지킬 수 없습니다.

모터의 `Present Input Voltage`(0.1 V 단위)를 1초에 한 모터씩 돌아가며 읽습니다. 걸을 때 전압이 순간적으로 떨어지므로 **8초 저역통과 값**으로 판단합니다.

| 단계 | 기본값 | 동작 |
|---|---|---|
| 주의 | < 7.0 V (3.5 V/셀) | 경고음 + 메시지 |
| 부족 | < 6.8 V (3.4 V/셀) | 걷는 중이면 멈추고, 새로 걷기 시작 금지 |
| 위험 | < 6.4 V (3.2 V/셀) | 10초 카운트다운 후 토크 OFF, 모터 LED 빨강, 토크 재가동 금지 |

- 런타임이 안 돌 때도 지키려면 `sudo bash install_robot.sh --battery-service`. 20초마다 확인해서 모든 SSH 터미널에 `wall` 경고, 모터 LED 빨강, **위험 전압이 60초 이어지면 오렌지파이 종료**. 단, **모터는 토크를 꺼도 대기 전류를 먹으니** 결국 전원 스위치를 꺼야 합니다.
- 서비스가 켜져 있으면 다른 모터 스크립트(`setup_motor.py` 등) 쓰기 전에 `sudo systemctl stop mero-battery`.
- 멀티미터로 배터리 단자 전압을 한 번 재서 모터 값과 다르면 `config.json`의 `battery.offset_v`로 맞추세요.
- 임계값은 `config.json`의 `battery`에서 바꿉니다.

## 4-1. 모터를 5V 레귤레이터로 공급하는 경우 (MERO 로봇 현재 구성)

원본 Microban은 모터에 2S 배터리(약 7.5V)를 그대로 넣고, `walk.onnx`도 그 전압으로 학습됐습니다. 모터를 5V로 공급하면 같은 P 게인에서 힘이 약 35% 약해집니다. 시뮬레이션에서는 기본 게인(서기 400)으로는 서 있지도 못했고, 게인을 7.6/4.9배(≈1.55배)로 올리면 7.5V일 때와 비슷하게 걸었습니다(30 Hz, 지연 18 ms).

또 모터가 읽는 전압은 레귤레이터 출력(4.9V)이라 **배터리 잔량을 알 수 없습니다.** 그래서 **INA219 전압 센서를 레귤레이터 앞(배터리 쪽)에** 달고, 배터리 보호는 이 센서 값으로 합니다.

### INA219 연결

| INA219 | 연결 |
|---|---|
| VCC | Orange Pi 3.3V |
| GND | Orange Pi GND (= 배터리 −, 공통 접지) |
| SDA / SCL | I2C1 SDA / SCL (IMU와 같은 버스, 주소가 달라 같이 써도 됨) |
| VIN+ , VIN− | 둘 다 **배터리 +** (레귤레이터 입력 쪽). 전류는 안 재고 전압만 재므로 션트에 직렬로 넣을 필요 없음 |

- 측정 범위 0~26V라 2S(최대 8.4V)는 분압 없이 바로 됩니다. 분압 저항을 썼다면 `sensor.scale`에 비율을 넣으세요.
- 로봇에서 `i2cdetect -y 1` → `40`(IMU의 `19`, `68`과 함께)이 보여야 합니다. A0/A1 점퍼로 주소를 바꿨으면 `sensor.addr`를 맞추세요.

### 설정

```bash
cp config.5v.example.json config.json
mero battery     # "배터리(ina219) 7.xx V" 가 나와야 함. 멀티미터 값과 다르면 battery.offset_v
```

[`config.5v.example.json`](config.5v.example.json)의 내용은 이렇습니다.
- `control`: 30 Hz, `kp_stand` 620, `kp_walk` 190(게인 약 1.55배)
- `battery.source`: `"ina219"`. 임계값은 2S 기준 그대로(7.0 / 6.8 / 6.4 V)

`run`, `calib`, `battery --watch`(서비스)가 모두 이 센서 값으로 경고하고, 걷기를 멈추고, 토크를 끕니다. 센서는 모터 버스를 쓰지 않으므로 런타임이 돌 때도 서비스가 계속 읽습니다.

센서가 없거나 응답이 없으면 **배터리 보호가 꺼진 상태**라고 빨간 경고를 띄우고(상태줄 "배터리 측정 불가"), 걷기는 막지 않습니다. 그동안에는 멀티미터로 직접 확인하세요. 모터 전압(4.9V)을 2S 기준으로 판단하면 늘 "위험"이 되므로, 센서 대신 모터 값으로 바꾸지는 않습니다.

센서를 달기 전 임시로만 쓰려면 `battery`를 `{"source": "motor", "cells": 1, "warn_v": 4.75, "low_v": 4.6, "critical_v": 4.4, "hysteresis_v": 0.05}`로 둘 수 있습니다. 이 값은 "레귤레이터가 못 버틸 만큼 배터리가 바닥났을 때"만 잡으므로 **리튬이온 과방전은 못 막습니다.**

> 레귤레이터 용량도 확인하세요. XL330 19개가 걸을 때 순간적으로 수 A를 먹습니다. 레귤레이터가 못 버티면 모터 전압이 떨어져 모터가 리셋되거나 넘어집니다. `mero battery`의 모터 값이 걷는 중 4.5V 아래로 자주 내려가면 더 큰 레귤레이터가 필요합니다.

## 5. 정책 바꾸기 (우리가 학습한 ONNX)

```bash
mero run --policy models/our_walk.onnx --gyro-frame body
```

**우리 보행 정책 `models/mero_walk_dr.onnx`** (`microban_rl` 의 `Microban-Walk-DR`, 루트 README "보행" 참고): walk.onnx 로 초기화한 뒤 이 로봇 조건(30 Hz, 5 V `kp_walk` 190 ≈ 0.275 Nm/rad, 지연 11–28 ms)에 바닥 마찰·강성 랜덤화와 기어 백래시(0–3°)·보정 오차(±2°)를 넣어 미세조정한 정책입니다. 관측·행동 형식은 walk.onnx 와 같고(63개, 18개), 자이로만 몸통 기준입니다.

```bash
mero run --policy models/mero_walk_dr.onnx --gyro-frame body     # config.json: control.hz 30, kp_walk 190
```

- mjlab에서 내보낸 ONNX의 메타데이터(`joint_names`, `default_joint_pos`, `action_scale`)를 읽어 관절 순서와 기본 자세를 맞춥니다. 관측 63개(위상 포함 65개) 형식이어야 합니다.
- **`gyro_frame`이 중요합니다.** 업스트림 `walk.onnx`는 MJCF `imu` 사이트 기준 자이로(`imu`)를 받습니다. 우리 MJLab 환경처럼 몸통 기준 `base_ang_vel`로 학습했다면 `body`.
- 학습 쪽 모터 게인이 다르면 `config.json`의 `control.kp_walk`(레지스터 값, 125 ≈ 0.28 Nm/rad)를 맞추세요.

## 6. 115200 bps의 한계

오렌지파이 UART로 1 Mbps가 안 돼서 115200을 씁니다. 그래서 통신이 가장 큰 병목입니다.

| 읽기 방식 | 1틱 바이트(18개 모터) | 이론 최대 |
|---|---|---|
| Sync Read (8바이트) | 478 | 24 Hz |
| Fast Sync Read (8바이트) | 360 | 32 Hz |
| **Fast Sync Read + Indirect(위치·속도 각 16비트)** | **288** | **40 Hz** |

런타임은 자동으로 가장 빠른 방식을 고릅니다(펌웨어 v46 이상 필요, `check`에서 확인). 전송 시간까지 흉내 낸 시뮬레이션에서 p95 약 30 ms(≈33 Hz)라 기본 `control.hz`를 **30**으로 뒀습니다. 실물 값은 `mero bench`로 확인하세요. `walk.onnx`는 50 Hz 학습이지만, 시뮬레이션 시험(아래)에서는 25~50 Hz, 지연 0~30 ms에서 모두 넘어지지 않고 걸었습니다. 실물에서 불안하면:
1. U2D2나 OpenRB-150을 USB로 꽂고 `mero set-baud 1000000` → `config.json`에 `{"serial": {"port": "/dev/ttyUSB0", "baud": 1000000}, "control": {"hz": 50}}`
2. 또는 MJLab에서 30 Hz(decimation)로 다시 학습

## 7. 로봇 없이 시험

노트북(또는 아무 리눅스)에서 `pip install mujoco onnxruntime numpy pyserial pytest` 후:

```bash
MERO_SIM=1 python -m mero run          # MuJoCo 로봇으로 같은 키보드 조종
MERO_SIM=held python -m mero calib     # 몸통을 공중에 고정한 채 캘리브레이션 UI
MERO_SIM=1 MERO_SIM_BAUD=115200 python -m mero bench
python -m pytest -q tests              # 39개 테스트
```

시뮬레이션은 실제 시리얼 패킷(CRC, Fast Sync Read, Indirect, 에코)까지 가짜 모터 체인으로 주고받으므로, 프로토콜·보정·IMU 좌표계·정책 관측·안전장치를 실물 없이 검증합니다. 단, 모터 모델은 단순화(XL330 전압 제어 근사)라 실물 거동과 같지는 않습니다.

## 8. 문제 해결

- **응답 없는 모터**: 전원 스위치, 데이지 체인, `mero check`로 어느 ID인지 확인. 115200이 아닌 모터는 안 보입니다.
- **Permission denied /dev/ttyAS0**: `dialout` 그룹 추가 후 재접속(또는 `sudo`).
- **ttyAS0에 이상한 글자/통신 오류**: 시리얼 콘솔이 붙어 있는지 확인(`systemctl status serial-getty@ttyAS0`).
- **제어 주기 경고**: `mero bench`의 권장값으로 `control.hz`를 낮추세요.
- **Gazebo 창이 검은색**(WSL): `--render-engine ogre` 또는 `export LIBGL_ALWAYS_SOFTWARE=1`.
- **뷰어가 연결 안 됨**: 로봇에서 `calib`/`run`이 돌고 있는지, 포트 9870, 같은 네트워크(노트북 핫스팟)인지 확인.

---

출처: 관절 이름·ID·부호·기본 자세·IMU 장착·`walk.onnx`·MJCF는 [Rhoban/microban](https://github.com/Rhoban/microban) (commit d0a128d, GPL-3.0, `LICENSE-microban`)에서 가져왔습니다.
