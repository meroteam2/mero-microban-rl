# Microban × MJLab (MuJoCo Warp) 강화학습 환경

[Rhoban/microban](https://github.com/Rhoban/microban) 휴머노이드(19 DoF, 0.82 kg)를
[MJLab](https://github.com/mujocolab/mjlab)(MuJoCo Warp 기반, GPU 병렬 시뮬레이션)에 올려
PPO(rsl-rl)로 학습하는 환경입니다.

- `Microban-Stand`: 넘어지지 않고 서 있기
- **`Microban-Crawl-Arms-2` (최종 포복)**: 엎드린 자세에서 다리와 팔을 대각선으로 번갈아 써서 기어가기
  → [포복 태스크](#포복-태스크-최종-microban-crawl-arms-2), 영상 `~/mero/videos/crawl.mp4`
- **`Microban-GetUp-5` + walk.onnx (일어나기 → 보행)**: 엎드린 자세에서 키프레임을 참조해 일어난 뒤 microban 의 walk.onnx 로 걷기
  → [일어나기 → 보행](#일어나기--보행-최종-microban-getup-5-model_2500--walkonnx), 영상 `~/mero/videos/getup_walk.mp4`
- **`Microban-Walk-DR2` (실물 MERO 보행)**: walk.onnx 로 actor 를 초기화하고 30 Hz·5 V 게인·바닥 DR·기어 백래시 환경에서 미세조정한 뒤, 실물 로그 분석에 따라 저전류로 다시 미세조정
  → [보행](#보행-최종-microban-walk-dr-walkonnx-초기화--미세조정), ONNX `mero_walk/models/mero_walk_dr2.onnx`

## 폴더 구조

```
~/mero/
├── README.md            ← 이 문서
├── LICENSE              ← GPL-3.0 (아래 "라이선스")
├── checkpoints/         ← 최종 체크포인트 (Stand, Crawl-Arms-2, GetUp-5, Walk-DR) + README
├── onnx/                ← 최종 정책 ONNX 3개 + getup_reference.npz + 사용법·실물 점검 README
├── videos/crawl.mp4     ← 최종 포복 정책 영상 (12 초, 비스듬한 추적 카메라)
├── videos/crawl_top.mp4 ← 같은 정책의 top 뷰 영상 (12 초)
├── videos/getup_walk.mp4 ← 일어나기 → 보행 영상 (20 초, 측면)
├── robot_keyframe_kit/  ← git clone (키프레임 에디터, commit 4d37f05)
├── keyframe_editor/     ← 에디터 실행 환경 (.venv), scene/config, init/, keyframes/*.lz4 (사용자 키프레임)
├── microban/            ← git clone (로봇 MJCF/URDF: src/model/mjcf/robot.xml)
├── mjlab/               ← git clone (editable 로 설치됨)
└── microban_rl/         ← 이 프로젝트 (uv 가상환경 .venv 포함)
    ├── pyproject.toml   ← 의존성, PyTorch cu130 인덱스 설정
    ├── src/microban_rl/
    │   ├── robot.py              ← Microban EntityCfg (MJCF 로드, XL330 액추에이터, 기본 자세)
    │   ├── stand_env_cfg.py      ← Microban-Stand 관측/행동/보상/종료 + PPO 설정
    │   ├── prone.py              ← 엎드린 시작 자세, 뒤집힘 판정
    │   ├── supine.py             ← 누운 시작 자세, 앞으로 구름 판정 (보류 중인 누운 일어나기용)
    │   ├── crawl_env_cfg.py      ← Microban-Crawl (= Crawl 3) 설정
    │   ├── crawl_arms_env_cfg.py ← Microban-Crawl-Arms / -Arms-2 (팔 보상, 모터 속도 제한)
    │   ├── crawl_cpg_env_cfg.py, cpg.py, curriculum.py, noise.py ← CPG 실험 (중단, 비교용 보존)
    │   ├── symmetry.py           ← 좌우 미러 대칭 (rsl-rl Symmetry 확장)
    │   ├── walk_onnx.py          ← microban walk.onnx 를 MJLab 에서 실행 (관측·지연·게인)
    │   ├── walk_env_cfg.py       ← Microban-Walk / -Walk-DR (실물 MERO 보행: 30 Hz, walk.onnx 관측 형식)
    │   ├── backlash.py           ← XL330 기어 백래시 + kp·토크·보정오차 랜덤화 액추에이터
    │   ├── floor_dr.py           ← 바닥 마찰·접촉 강성 랜덤화 (로봇 충돌 geom priority 1)
    │   ├── getup_env_cfg.py      ← Microban-GetUp (참조 residual 행동, RSI, 보상, 단계 판정)
    │   ├── getup2~5_env_cfg.py   ← run 2~5 변경점 (비틀기 제한, 손 떼기, 몸통 비틀기·한 팔·하중, pitch 중심)
    │   ├── video.py              ← 정책 영상 저장 (microban-record)
    │   ├── __init__.py           ← 태스크 등록
    │   └── train.py / play.py    ← mjlab train/play CLI 래퍼
    ├── scripts/
    │   ├── check_gpu.py          ← PyTorch·Warp GPU 동작 확인
    │   ├── check_robot.py        ← CPU MuJoCo로 로봇 로드/자세 확인
    │   ├── monitor_arms.py       ← 포복 정책 환경별 점검 (대칭·접촉·협응·속도)
    │   ├── walk_onnx_check.py    ← walk.onnx 시작 조건 스윕
    │   ├── make_keyframe_editor_files.py ← 키프레임 에디터 scene/config/초기 키프레임 생성
    │   ├── replay_keyframes.py   ← 키프레임 open-loop 재생 (토크·속도 한계)
    │   ├── make_getup_reference.py ← 학습용 키프레임 복사본 + 참조 데이터 (data/)
    │   ├── monitor_getup.py, measure_push.py ← 일어나기 환경별 점검 (단계·대칭·밀기·착지 충격)
    │   ├── getup_walk.py, record_getup_walk.py ← 4단계 전환 측정·viser / 영상
    │   ├── export_onnx.py        ← 최종 정책 ONNX 내보내기 (메타데이터, torch 대비 검증)
    │   ├── init_from_walk_onnx.py ← walk.onnx 가중치로 Walk 체크포인트 만들기 (자이로 프레임 변환)
    │   ├── export_walk_onnx.py   ← 보행 정책 평가 (명령별 넘어짐·속도, 밀기/최악 DR) + mero_walk 용 ONNX
    │   ├── check_walk_env.py, probe_walk.py ← 보행 환경 점검 (관측 형식, DR 적용) / 한 env 추적 출력
    │   ├── walk_pitch.py         ← 걷는 중 피치·모터 전류·관절 추종 (실물 로그와 비교용)
    │   ├── make_getup_back_init.py ← 누운 일어나기 에디터 시작 파일 (init/getup_back_init.lz4)
    │   └── analysis/             ← 일회성 분석 스크립트 (엎드린/누운 가능성, 키프레임 자세 계산, 대칭 점검)
    ├── data/                     ← 일어나기 학습용 키프레임 복사본, 참조 궤적 (getup_reference.npz)
    │   └── diagnose_rewards.py, monitor_symmetry.py, ... ← CPG 실험용 진단
    └── logs/rsl_rl/<실험>/<날짜>/  ← 체크포인트(model_*.pt), tensorboard 로그
```

## 검증된 버전 (2026-10-30)

| 항목 | 버전 |
| --- | --- |
| OS | Windows + WSL2 Ubuntu (kernel 6.18), Windows 쪽 NVIDIA 드라이버 610.88 |
| GPU | RTX 5080 16GB (Blackwell, sm_120) |
| Python | 3.12 (uv 가 관리) |
| PyTorch | 2.14.1+cu130 (CUDA 13.0 빌드, sm_120 커널 포함) |
| warp-lang | 1.18.0 (NVIDIA 인덱스) |
| mujoco / mujoco-warp | 3.16.0 / 3.16.0 |
| mjlab | 1.6.0 (commit `8bb2bc9`) |
| rsl-rl-lib | 5.5.1 |
| microban | commit `d0a128d` |

> WSL2 에서는 Windows 의 NVIDIA 드라이버를 그대로 사용합니다. **Linux 용 NVIDIA 드라이버나
> CUDA 툴킷을 apt 로 설치하지 마세요.** PyTorch/Warp 휠에 필요한 CUDA 런타임이 포함되어 있습니다.

## 처음부터 설치하기

```bash
# 1) uv 설치 (사용자 영역 ~/.local/bin, sudo 불필요)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env        # 또는 새 터미널 열기

# 2) 레포 받기 — 이 레포 (mero-microban-rl) 를 ~/mero 에 서브모듈과 함께
#    microban, mjlab, robot_keyframe_kit 은 서브모듈로 아래 커밋에 고정되어 있음
git clone --recursive https://github.com/meroteam2/mero-microban-rl.git ~/mero
#    (이미 받았다면: git -C ~/mero submodule update --init)
#    서브모듈 없이 직접 받는 경우:
#      git clone https://github.com/Rhoban/microban.git && git -C microban checkout d0a128d
#      git clone https://github.com/mujocolab/mjlab.git && git -C mjlab checkout 8bb2bc9
#      git clone https://github.com/Stanford-TML/robot_keyframe_kit.git && git -C robot_keyframe_kit checkout 4d37f05

# 3) 가상환경 생성 + 의존성 설치
cd ~/mero/microban_rl
uv python install 3.12
uv sync --python 3.12          # .venv 생성, torch(cu130)·warp·mjlab(editable) 설치

# 4) GPU 확인
uv run python scripts/check_gpu.py
#   torch arch list 에 'sm_120' 이 있고, matmul ok / kernel ok 가 나오면 정상
```

`pyproject.toml` 의 핵심 설정:

- `torch` → `https://download.pytorch.org/whl/cu130` 인덱스 (RTX 50 시리즈 지원 빌드)
- `warp-lang` → `https://pypi.nvidia.com/` 인덱스
- `mjlab` → `../mjlab` 로컬 클론을 editable 설치 (mjlab 소스를 고치면 바로 반영)

외부 레포 (서브모듈, 수정 없이 사용):

| 폴더 | 레포 | 커밋 | 라이선스 |
| --- | --- | --- | --- |
| `microban/` | https://github.com/Rhoban/microban | `d0a128d180ecfce69eaaf31b295a0fce31280bfd` | 소프트웨어 GPL-3.0, CAD CC BY-NC-SA 4.0 |
| `mjlab/` | https://github.com/mujocolab/mjlab | `8bb2bc9e187797689ec82c78c35019afd2fd0376` | Apache-2.0 |
| `robot_keyframe_kit/` | https://github.com/Stanford-TML/robot_keyframe_kit | `4d37f055134db883b5742b6c34c8518cb81a11bc` | MIT |

키프레임 에디터 (선택):

```bash
cd ~/mero/keyframe_editor && uv sync      # robot_keyframe_kit(editable) + mujoco 3.16.0, 별도 .venv
# microban_scene.xml / config.yaml 에는 이 PC 의 절대 경로(/home/mero/mero/...)가 들어 있으므로 다른 PC 에서는 다시 생성:
cd ~/mero/microban_rl && uv run python scripts/make_keyframe_editor_files.py
```

## 실행 명령어

모든 명령은 `~/mero/microban_rl` 에서 실행합니다. (`uv run` 이 자동으로 `.venv` 를 사용)

```bash
cd ~/mero/microban_rl

# 학습 (기본: 4096 env, 1500 iteration, 50 iter 마다 체크포인트 저장)
uv run microban-train Microban-Stand

# 짧게 동작 확인만
uv run microban-train Microban-Stand --agent.max-iterations 5

# 옵션 예시: env 수 / iteration 수 변경
uv run microban-train Microban-Stand --env.scene.num-envs 2048 --agent.max-iterations 500

# 학습 이어하기
uv run microban-train Microban-Stand --agent.resume True \
    --agent.load-run 2026-10-30_11-37-20 --agent.load-checkpoint model_299.pt

# 전체 옵션 보기
uv run microban-train Microban-Stand --help
```

학습 결과 확인:

```bash
# 학습된 정책 재생 → 브라우저에서 http://localhost:8080 접속 (viser 웹 뷰어)
uv run microban-play Microban-Stand \
    --checkpoint-file logs/rsl_rl/microban_stand/2026-10-30_11-37-20/model_299.pt \
    --viewer viser

# 정책 없이 (0 행동 / 랜덤 행동) 로봇만 보기
uv run microban-play Microban-Stand --agent zero --viewer viser

# TensorBoard
uv run tensorboard --logdir logs/rsl_rl/microban_stand   # → http://localhost:6006
```

`--viewer native` 를 쓰면 MuJoCo 네이티브 창이 WSLg 로 뜹니다. 창이 안 뜨면 `viser` 를 쓰세요.

## 태스크 설계 (`Microban-Stand`)

**로봇** (`robot.py`)
- microban 레포의 `src/model/mjcf/robot.xml` 을 그대로 로드 (`MICROBAN_ROOT` 환경변수로 경로 변경 가능)
- XML 의 토크 `<motor>` 는 제거하고 Dynamixel XL330 위치 제어를 흉내 내는 position 액추에이터로 교체
  - kp = 0.88 Nm/rad (`constants.py` 의 `KP_DEFAULT=400` × 0.0022), 토크 한계 0.64 Nm (1.75 A × 0.366 Nm/A)
- 기본 자세: microban 의 `walk.onnx` 메타데이터 `default_joint_pos` 값
- 초기 높이 0.172 m (발바닥이 바닥에 딱 닿는 높이)
- 참고: 정책 없이 PD 로 기본 자세만 유지하면 1초 안에 뒤로 넘어집니다 → 정책이 균형을 잡아야 함

**MDP** (`stand_env_cfg.py`, 시뮬 200 Hz / 정책 50 Hz, 에피소드 20 초)
- 관측(actor): 몸통 각속도, 중력 투영 벡터, 관절 위치/속도(기본 자세 기준), 이전 행동 (+노이즈)
- 관측(critic): actor 관측 + 몸통 선속도 + 몸통 높이 (특권 정보)
- 행동: 19 관절 목표 위치 = 기본 자세 + 0.25 × action
- 보상

  | 항 | 가중치 | 내용 |
  | --- | --- | --- |
  | alive | +1.0 | 살아있는 매 스텝 |
  | upright | +1.0 | exp(-기울기²/0.3²) |
  | base_height | +1.0 | exp(-(z-0.170)²/0.02²) |
  | fall | -50 | 넘어져서 종료되면 |
  | default_pose | -0.5 | 기본 자세와의 관절 오차² |
  | joint_vel / torques / action_rate | -1e-3 / -0.1 / -0.05 | 부드러운 동작 유도 |
  | joint_limits | -5.0 | 관절 한계(soft) 초과 |

- 종료: 기울기 > 0.8 rad, 몸통 높이 < 0.10 m, 시간 초과
- 외란: 리셋 시 위치/yaw/관절 랜덤화, 2~4 초마다 몸통에 ±0.3 m/s 속도 밀기

보상/종료 조건을 바꾸려면 `stand_env_cfg.py` 의 `rewards` / `terminations` 딕셔너리를 수정하면 됩니다.

## 확인된 학습 결과

RTX 5080, 4096 env 기준 약 15~22만 step/s. 300 iteration (≈ 2분 20초) 결과:

| iteration | 평균 보상 | 평균 에피소드 길이 (최대 1000) |
| ---: | ---: | ---: |
| 0 | -1.07 | 24 |
| 50 | 23.3 | 734 |
| 100 | 40.2 | 961 |
| 200 | 54.3 | 995 |
| 299 | 58.2 | 998 (넘어짐 0) |

## 포복 태스크 (최종: `Microban-Crawl-Arms-2`)

엎드린 자세에서 시작해 몸통을 월드 +x 방향으로 낮게 기어가는 태스크입니다. 참조 동작 없이 PPO만 썼고,
최종 정책은 **`logs/rsl_rl/microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt`** 입니다.

### 학습 과정

| 단계 | 태스크 | 시작점 | iteration | 학습 시간 | 내용 |
| --- | --- | --- | ---: | ---: | --- |
| 1. Crawl 3 | `Microban-Crawl` | 처음부터 | 0 → 1499 | 13분 30초 | 다리만 좌우 교대로 쓰는 대칭 포복 (팔 미사용) |
| 2. Crawl-Arms | `Microban-Crawl-Arms` | Crawl 3 `model_1499` | 1499 → 3498 | 19분 44초 | 팔 보상 추가, 모터 속도 제한 10.3 rad/s 적용. 다리는 유지됐지만 팔이 한 번도 바닥에 안 닿음 |
| 3. Crawl-Arms-2 | `Microban-Crawl-Arms-2` | Crawl-Arms `model_3498` | 3498 → 5497 | 31분 50초 | 아래팔 유도 보상, 팔 자세 비대칭 페널티 지연, 팔 관절 탐색 노이즈 증가 → 대각선 협응 포복 |

합계 약 65분 (RTX 5080, 4096 env). 단계별 실패 원인:
- **Crawl-Arms**: Crawl 3 정책의 행동 노이즈가 작고(std 0.14~0.16), 아래팔이 바닥에서 평균 27 mm 떠 있음.
  그래서 접촉 보상이 거의 발생하지 않았고, 신호가 있던 팔 자세 비대칭 페널티는 오히려 팔 탐색을 막았음.
- **Crawl-Arms-2**: 위 세 가지를 고쳐서 iteration ~4200에 양팔 접촉이 10%를 넘었음. 그 뒤 비대칭 페널티가 켜지고 접촉이 42%까지 증가.

### 최종 보상 설계 (`crawl_env_cfg.py` + `crawl_arms_env_cfg.py`)

공통 설정 (Crawl 3 그대로):
- 시뮬 200 Hz / 정책 50 Hz, 에피소드 10 초
- 행동: 19 관절 목표 = 엎드린 기본 자세 + 0.5 × action
- 관측: 각속도, 중력 투영, heading(sin, cos), 관절 위치·속도, 이전 행동 (+critic: 선속도, 몸통 높이)
- 종료: 뒤집힘(belly_up), 시간 초과
- 좌우 대칭: 미러 데이터 증강 + 미러 손실 (`symmetry.py`)

| 항 | 가중치 | 내용 |
| --- | ---: | --- |
| forward_vel | +10 | 몸통 +x 속도, 0.2 m/s 에서 상한 |
| trunk_low | +1 | 몸통 원점 높이 < 0.06 m |
| contact_slip | −4 | 바닥에 닿은 모든 부위(팔 포함)의 미끄럼 속도² 합 → 끌지 말고 짚고 밀기 |
| contact_force | −0.5 | 부위별 수직 접촉력이 체중 1.5배를 넘는 만큼 |
| touchdown_speed | −1 | 바닥 직전(1.5 cm 이내) 부위의 하강 속도² |
| heading / lateral_vel / yaw_rate | −1 / −5 / −0.2 | 직진 유지 |
| torques / action_rate / joint_limits | −0.05 / −0.02 / −5 | 부드러운 동작, 관절 한계 |
| flipped | −20 | 뒤집혀 종료 |
| **arm_contact** | +0.3 | 바닥에 닿은 팔 비율 (아래팔 어느 부위든) |
| **arm_leg_diagonal** | +1.0 | 대각선 쌍(왼팔+오른다리, 오른팔+왼다리)이 함께 접지 |
| **arm_leg_ipsilateral** | −0.5 | 같은 쪽 쌍이 함께 접지 |
| **arm_contact_balance** | −1.0 | 최근 ~1.5 초 좌우 팔 접촉 비율 차이 |
| **arm_pose_asym** | −1.0 | 최근 ~1.5 초 좌우 팔 관절 평균 자세 비대칭. 배치 평균 팔 접촉이 좌우 각 10%를 넘은 뒤 100 iteration에 걸쳐 켜짐 |
| **arm_floor_guide** | +0.5 | 팔마다 exp(−아래팔 최저점 높이 / 2 cm). 대각선 다리가 접지 중이면 1, 아니면 0.25 배. 500 iteration에 걸쳐 10%→100% |

- **모터 속도**: 관절 목표가 한 정책 스텝에 XL330 무부하 속도(123 rpm @ 6 V = 12.9 rad/s)의 80%, 즉 10.3 rad/s 이상 움직이지 않도록 제한합니다 (`SpeedLimitedJointPositionAction`).
  - Crawl 3은 큰 목표 점프에 의존해서, 이 제한을 한 번에 걸면 0.21 → 0.04 m/s 로 떨어집니다.
  - 그래서 Crawl-Arms 학습 초반 400 iteration 동안 60 → 10.3 rad/s 로 서서히 좁혔습니다. Crawl-Arms-2 는 처음부터 10.3 rad/s 입니다.
- **탐색 노이즈**: Crawl-Arms-2 시작 시 팔 6개 관절의 행동 std 를 0.4 로 올렸습니다 (`ArmNoiseRunner`). 다리 관절은 그대로입니다.

### 결과 비교 (같은 기준)

평가 기준:
- 재생 설정, 128 env, 10 초 중 뒤 5 초.
- **좌우 비대칭**은 환경별로 |q_L − 부호·q_R| (관절별 평균 자세)를 구한 뒤 env 간 중앙값/p90.
- CPG 1~4 는 학습 당시 CPG 설정으로 평가.

| 버전 | 전진 속도 m/s | 비대칭 중앙값 / p90 rad | 팔꿈치 접촉 L / R | 손 접촉 L / R | 발 접촉 L / R | 최고 관절 속도 최대 / p99 rad/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crawl 1 | 0.296 | 0.495 / 0.497 | 0 / 0% | 60 / 56% | 53 / 22% | 12.4 / 11.0 |
| Crawl 2 | 0.237 | 0.016 / 0.020 | 0 / 0% | 0 / 0% | 20 / 20% | 12.3 / 10.7 |
| Crawl 3 | 0.210 | 0.003 / 0.006 | 0 / 0% | 0 / 0% | 44 / 45% | 11.3 / 9.5 |
| CPG 1 | −0.009 | 0.180 / 0.222 | 0 / 0% | 52 / 46% | 6 / 4% | 13.4 / 8.7 |
| CPG 2 | 0.001 | 0.795 / 0.802 | 0 / 0% | 0 / 0% | 0 / 0% | 13.9 / 7.0 |
| CPG 3 | 0.084 | 0.046 / 0.084 | 18 / 21% | 19 / 20% | 0 / 0% | 15.0 / 12.5 |
| CPG 4 | 0.192 | 0.799 / 0.801 | 0 / 37% | 39 / 0% | 17 / 55% | 13.9 / 12.4 |
| CPG 5 | 0.176 | 0.544 / 0.546 | 0 / 0% | 63 / 61% | 21 / 0% | 11.7 / 10.1 |
| Crawl-Arms | 0.151 | 0.003 / 0.004 | 0 / 0% | 0 / 0% | 44 / 44% | 10.7 / 9.1 |
| **Crawl-Arms-2 (최종)** | **0.191** | **0.002 / 0.003** | **44 / 44%** | 0 / 0% | 48 / 48% | 11.6 / 11.0 |

최종 정책의 팔-다리 협응 (`scripts/monitor_arms.py`, 16 초 중 뒤 12 초, 환경별 중앙값):
- 팔이 짚는 동안 **대각선 다리도 접지 88%**, 같은 쪽 다리는 12%.
- 팔 접촉과 대각선 다리 접촉의 위상차는 17° (0° = 완전히 같은 박자).
- 팔·다리 주기는 모두 1.17 Hz. 69% 환경에서 주된 주파수가 일치합니다 (나머지는 팔이 한 주기에 두 번 닿는 것으로 보임).
- **팔의 전진 기여**:
  - 전체 앞 방향(+x) 지면 밀기 중 22% 가 팔에서 나옵니다.
  - 평균 알짜 +x 힘은 팔 +0.32 W (W = 체중), 나머지(다리+몸통) −0.32 W 입니다.
- 접지 중 팔 미끄럼 속도는 0.056 m/s, 좌우 팔 접촉 비율 차이는 0.01 입니다.
- 팔은 손끝이 아니라 **팔꿈치 쪽 원통**으로 짚습니다. 짚는 부위는 제한하지 않았습니다.

### 실행 · 재생 · 영상 저장

```bash
cd ~/mero/microban_rl

# 1) Crawl 3 (처음부터, 1500 iteration)
uv run microban-train Microban-Crawl

# 2) Crawl-Arms: Crawl 3 체크포인트에서 이어서 2000 iteration
#    (load-run 은 같은 실험 폴더 안에서 찾으므로 체크포인트를 복사해 둠)
mkdir -p logs/rsl_rl/microban_crawl_arms/crawl3_init
cp logs/rsl_rl/microban_crawl/2026-10-30_12-46-39/model_1499.pt logs/rsl_rl/microban_crawl_arms/crawl3_init/
uv run microban-train Microban-Crawl-Arms --agent.resume True \
    --agent.load-run crawl3_init --agent.load-checkpoint model_1499.pt

# 3) Crawl-Arms-2: Crawl-Arms 체크포인트에서 이어서 2000 iteration
uv run microban-train Microban-Crawl-Arms-2 --agent.resume True \
    --agent.load-run 2026-11-04_11-16-02 --agent.load-checkpoint model_3498.pt

# 학습 중 환경별 점검 (500 iteration 마다, 비대칭 > 0.1 rad 또는 속도 < 0.1 m/s 면 학습 중단)
uv run python scripts/monitor_arms.py logs/rsl_rl/microban_crawl_arms/<run> \
    --task-id Microban-Crawl-Arms-2 --train-pid <학습 PID> --first 4000 --last 5497
# 체크포인트 하나만 점검
uv run python scripts/monitor_arms.py --task-id Microban-Crawl-Arms-2 \
    --checkpoint-file logs/rsl_rl/microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt

# 재생 (viser) → http://localhost:8080
uv run microban-play Microban-Crawl-Arms-2 --viewer viser \
    --checkpoint-file logs/rsl_rl/microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt

# 영상 저장 (12 초, 1280x720, 50 fps)
uv run microban-record Microban-Crawl-Arms-2 \
    --checkpoint-file logs/rsl_rl/microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt \
    --output ~/mero/videos/crawl.mp4 --seconds 12

# 위에서 본 top 뷰 영상 (카메라가 바로 위에서 내려다봄, 오른쪽이 진행 방향 +x)
uv run microban-record Microban-Crawl-Arms-2 \
    --checkpoint-file logs/rsl_rl/microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt \
    --output ~/mero/videos/crawl_top.mp4 --seconds 12 --elevation -89.9 --azimuth 90

# TensorBoard
uv run tensorboard --logdir logs/rsl_rl/microban_crawl_arms
```

이전 버전 체크포인트 (모두 보존):
- `logs/rsl_rl/microban_crawl/`
  - `2026-10-30_12-05-24` = Crawl 1
  - `12-29-35` = Crawl 2
  - `12-46-39` = Crawl 3
- `logs/rsl_rl/microban_crawl_cpg/` = CPG 1~5
- `logs/rsl_rl/microban_crawl_arms/2026-11-04_11-16-02` = Crawl-Arms

## 일어나기 → 보행 (최종: `Microban-GetUp-5` model_2500 + walk.onnx)

엎드린 자세에서 일어나 기본 자세로 선 뒤, microban 레포의 `walk.onnx` 보행 정책으로 넘겨 걷게 합니다.
영상: `~/mero/videos/getup_walk.mp4` (20 초, 측면, 일어나기 13.9 초 → 정지 1 초 → 보행 5.1 초).

### 단계별 진행

| 단계 | 내용 | 결과 |
| --- | --- | --- |
| 1. walk.onnx 재현 | `walk_onnx.py`: ONNX 가중치를 torch로 옮겨 MJLab에서 배치 실행 (IMU 프레임 자이로, 센서 지연 1/3/4 tick, kp 0.275) | vx 0.15 → 0.146 m/s, 성공 100%. 시작 조건: \|roll\| < 0.2, pitch −0.1 ~ +0.5, 관절 RMS < 0.12, 양발 바닥, 거의 정지 |
| 2. 키프레임 | robot_keyframe_kit 에디터로 직접 제작 (아래) | `keyframes/getup_v6.lz4` |
| 2-1. open-loop 재생 | `replay_keyframes.py`: 토크 한계 0.64 Nm + 속도 제한 10.3 rad/s | pike_push 까지는 따라가지만 손→발 무게 이동에서 실패 (PD만으로는 0.4 Nm 토크를 못 냄) |
| 3. 학습 | `Microban-GetUp` run 1~5 (아래) | run 5 model_2500: 성공 100% |
| 4. 전환 | `getup_walk.py` | 전환 성공 100% (32 env) |

### 키프레임 제작 (robot_keyframe_kit)

- 설치: `~/mero/robot_keyframe_kit` (commit `4d37f05`), 실행 환경 `~/mero/keyframe_editor/.venv`
  (viser-keyframe 포크가 mjlab 의 viser 를 덮어쓰므로 분리, mujoco 3.16.0 고정).
- 에디터 파일은 `scripts/make_keyframe_editor_files.py` 가 생성: `microban_scene.xml` (토크 한계 0.64 Nm, MJLab 과 같은
  물리 설정, 손 IK 사이트), `config.yaml` (symmetry.py 의 미러 쌍·부호 — 에디터 자동 부호는 pitch 관절 5개가
  틀려 미러 자세가 최대 194 mm 어긋남), `init/getup_init.lz4` (prone, stand_target).
- 에디터는 포트 8081 (`run_editor.py`, viser play 의 8080 과 동시 사용 가능).
- 사용자가 만든 `getup_1111.lz4` 를 open-loop 로 재생하니 pike_1 → crouch 에서 앞으로 엎어짐: 무게중심이 발끝보다
  10 cm 앞인 채로 손이 먼저 떨어짐. 무게중심을 발 위에 둔 채로는 손이 바닥에 닿지 않아 (최소 22 mm 위, 고관절·무릎·
  발목 모두 한계 근처) "손으로 밀어 넘기기"가 필수 → 중간 자세 3개를 넣어 `getup_v6.lz4` 를 만듦.

`getup_v6` 시퀀스 (학습용 복사본 `microban_rl/data/getup_v6_train.lz4` 는 pike_push → fold 를 0.8 초로 늘린 것):

| 시간 (학습용) | 자세 | 역할 |
| ---: | --- | --- |
| 0.0 | prone | 엎드린 시작 자세 |
| 1.5 | arm_prep | 팔꿈치를 굽혀 손을 가슴 아래로 |
| 2.5 | chest_up | 팔로 가슴을 들어 올림 |
| 3.5 | plank | 손·발로 몸통을 들어 플랭크 |
| 5.0 | plank_1 | 고관절·무릎을 굽혀 다리를 당길 준비 |
| 7.0 | pike | 엉덩이를 높이 (손·발 4점 지지) |
| 8.0 | pike_1 | pike 미세 조정 — 무게중심이 발끝 12 mm 앞 |
| 8.6 | **pike_push** | 손으로 밀어 엉덩이를 발 위로 (무게중심이 발끝 6.6 mm 앞, 손을 짚은 채 가능한 최대) |
| 9.4 | **fold** | 손을 떼고 발로만 섬, 다리 편 채 상체를 숙인 자세 (무게중심이 발끝 20 mm 뒤) |
| 10.4 | **squat_deep** | 무릎을 굽히며 상체를 세움 (몸통 기울기 27°) |
| 11.4 | crouch | 웅크려 앉은 자세 |
| 12.4 | stand_target | 1단계에서 승인한 시작 자세 (기본 관절, pitch +0.175, 양발 평평) |

### 학습 과정 (run 1 → 5)

모두 `logs/rsl_rl/microban_getup/` 에 보존. 각 run 은 앞 run 의 체크포인트에서 이어서 학습했습니다.

| run | 시작점 | 생긴 문제 | 고친 점 (다음 run) |
| --- | --- | --- | --- |
| 1 `2026-11-16_19-49-21` | 처음부터 | it 500 에서 63% 가 손 밀기를 통과했지만 밀기 구간에서 **한쪽으로 비틀기** (hip_yaw 좌우 차 0.82 rad), it 1000 에서 퇴보 | 밀기 구간 자세 대칭 페널티, 정규화 램프 완만히, fold/squat_deep 시작 늘림, 비틀기 관절 residual ±0.1 rad 제한 |
| 2 `..._20-08-04_getup2` | run 1 m500 | 비틀기는 줄었지만 (0.32 → 0.20) **손 짚은 채 pike_push 에 머묾** (it 1500: 93%) | fold 시점 이후 손 접촉 페널티, weight_transfer 2 → 4, fold 시작 4배, 넓은 imitation 허용폭을 fold 까지만 |
| 3 `..._20-30-14_getup3` | run 2 m1000 | **성공 100%** 이지만 **한 팔로만 지탱**하며 몸통이 옆으로 크게 비틀림 (roll 42°, yaw 85°, 한 손만 짚는 시간 95%; 좌/우 팔 방향은 env 마다 거울상으로 갈림) — 관절각 지표로는 안 보였음 | 밀기 구간 몸통 roll² + yaw 변화² 페널티, 한 손만 짚기 페널티, 좌우 팔 하중 차 페널티 |
| 4 `..._20-47-08_getup4` | run 3 m1500 | 밀기는 대칭이 됨 (roll 3°, 양손 동시) 대신 짧게 뛰어 착지하는 방식으로 바뀌고, 선 자세가 **pitch 0.200 (허용 범위 0 ~ 0.2 의 경계)** 에 머물러 1초 유지 판정이 거의 실패 | 선 자세 pitch 0.175 중심 보상 |
| **5** `..._21-07-16_getup5` | run 4 m2000 | — | **최종: model_2500** |

run 3 의 성공 체크포인트(model_1500, model_2000)는 `logs/rsl_rl/microban_getup/video_backup_getup3/` 에도 백업.

### 최종 보상 설계 (`getup_env_cfg.py` + `getup2~5_env_cfg.py`)

- 행동: 참조 궤적 기준 residual — 목표 관절각 = q_ref(t+dt) + 0.5 × action, 모터 속도 제한 10.3 rad/s,
  토크 한계 0.64 Nm, kp 0.88. 밀기 구간(pike_1 → squat_deep)에서 hip_yaw·hip_roll·ankle_roll residual 은 ±0.1 rad.
- 관측: IMU 각속도, 투영 중력, 관절 위치·속도, 직전 행동, 위상(참조 시간 / 끝 시간), 다음 참조 관절각.
- 시작: 40% 엎드림(prone.py 랜덤화), 25% open-loop 재생 상태(pike_push 까지), 35% 키프레임 자세 (fold 4배, squat_deep 2배).
- 종료: 등이 하늘로 뒤집힘(belly_up), 참조 끝 + 3 초. 넘어짐 종료 없음. 좌우 미러 증강 + 미러 손실.

| 항 | 가중치 | 내용 |
| --- | ---: | --- |
| pose_imitation | 1.0 | exp(−관절 오차² / σ²), σ 0.3 rad (pike_1 → fold 구간 3배) |
| trunk_imitation | 1.0 | 몸통 기울기·높이 참조 추종 (같은 구간 3배) |
| push_shaping | 50 | 포텐셜 기반: 캡처 포인트가 발바닥 밖으로 벗어난 거리의 감소 |
| weight_transfer | 4.0 | pike_push 이후 손 떼고 평평한 발바닥만으로 서 있고 캡처 포인트가 발바닥 안 |
| upright_height | 1.0 | squat_deep 이후 cos(기울기) × 높이 |
| stand_target | 2.0 | 선 자세 조건 1, 1초 유지 후 3 |
| stand_pitch | 1.0 | crouch 이후 exp(−(pitch − 0.175)² / 0.05²) |
| push_symmetry | −1.0 | 밀기 구간 좌우 관절각 차이 |
| trunk_twist | −3.0 | 밀기 구간 몸통 roll² + yaw 변화² |
| one_hand | −2.0 | 밀기 구간 한 손만 바닥에 닿음 |
| hand_load_diff | −0.5 | 밀기 구간 좌우 팔 수직력 차 / 체중 |
| hands_after_fold | −1.0 | fold 시점 이후 손이 바닥에 닿음 |
| face_plant / knee_contact | −2.0 / −0.3 | pike_1 이후 몸통·머리 / 무릎 접촉 |
| torques, action_rate, joint_limits, contact_slip | −0.05, −0.02, −5, −1 | 정규화 (run 1~4 에서 0.1 → 1 로 램프, run 5 는 1) |
| flipped | −20 | 뒤집힘 종료 |

### 결과 비교 (엎드린 시작, 성공 = 선 자세 조건 1초 유지)

| 체크포인트 | 성공률 | 끝 pitch (rad) | 밀기 몸통 roll / yaw (최대, 중앙값) | 한 손만 짚기 | 왼팔 하중 비율 | 밀기 hip_yaw 좌우 차 | 착지 충격 (한 발 최대, 체중 대비) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| run 1 m500 | 0% (squat_deep 35%) | — | — | — | — | 0.82 rad | — |
| run 2 m1500 | 0% (pike_push 93%) | — | — | — | — | 0.20 rad | — |
| run 3 m1500 | 100% | +0.175 (m2000) | 42° / 85° | 95% | 한 팔 99% | 0.24 rad | — |
| run 4 m2000 | 0% (경계 pitch) | +0.200 | 3.0° / 8.3° | 2% | 46% | 0.025 rad | 3.0 (p90 3.5) |
| **run 5 m2500** | **100%** | **+0.176** | **0.9° / 2.9°** | **1%** | **53%** | **0.012 rad** | **3.0 (p90 3.5)** |

run 5 m2500: 일어나는 시간 12.44 초, 뒤집힘 0%, 최대 토크 0.64 Nm (토크 한계에 닿는 시간 0.1%), 최대 관절 속도 8.4 rad/s,
walk.onnx 시작 조건 만족 100%. 착지 충격 3.0 × 체중은 walk.onnx 보행 중 발 최대 수직력 (3.2 × 체중) 과 같은 수준.

### 4단계: 일어나기 → 보행 전환 (`scripts/getup_walk.py`)

1. 일어나기 정책으로 참조 끝 + 1 초까지.
2. **1 초 정지**: kp 0.88 로 기본 자세(stand_target 관절)를 유지하되, 일어나기 정책이 그 자세 주변에서 균형을 계속 잡음
   (`--hold-mode policy`, 기본값). 고정 목표만 주는 `--hold-mode pd` 는 kp 0.88 로 기본 자세를 버티지 못하고 1 초 안에
   뒤로 넘어짐 (전환 성공 0%).
3. MJLab `pd_gains` 로 env 별 게인을 **0.275** 로 바꾸고 walk.onnx (센서 지연 포함) 로 vx 0.15, 약한 heading hold
   vθ = −1.0 × (전환 후 방향 변화), |vθ| ≤ 0.3.

결과 (model_2500, 32 env): 전환 성공 100%, 전환 시 pitch +0.177, 보행 중 넘어짐 0%, 0.131 m/s, 6 초 동안 옆으로 −0.02 m,
방향 +4.7°, 한 발 약 9 걸음. heading hold 없이는 6 초에 +16.6° (run 4 m2000 기준), 0.5 면 +6.7°.

### 실행 · 재생 · 영상 저장

```bash
cd ~/mero/microban_rl

# 키프레임 에디터 (http://localhost:8081)
cd ~/mero/keyframe_editor && uv run python run_editor.py --data keyframes/getup_v6.lz4 && cd -

# 키프레임 open-loop 재생 (토크·속도 한계, 구간별 토크 포화)
uv run python scripts/replay_keyframes.py ~/mero/keyframe_editor/keyframes/getup_v6.lz4

# 학습용 참조 데이터 생성 (data/getup_v6_train.lz4, data/getup_reference.npz)
uv run python scripts/make_getup_reference.py

# 학습: run 1 → run 5 (각각 앞 run 체크포인트에서 이어서)
uv run microban-train Microban-GetUp
uv run microban-train Microban-GetUp-2 --agent.resume True --agent.load-run 2026-11-16_19-49-21 --agent.load-checkpoint model_500.pt
uv run microban-train Microban-GetUp-3 --agent.resume True --agent.load-run 2026-11-16_20-08-04_getup2 --agent.load-checkpoint model_1000.pt
uv run microban-train Microban-GetUp-4 --agent.resume True --agent.load-run 2026-11-16_20-30-14_getup3 --agent.load-checkpoint model_1500.pt
uv run microban-train Microban-GetUp-5 --agent.resume True --agent.load-run 2026-11-16_20-47-08_getup4 --agent.load-checkpoint model_2000.pt

# 학습 중 환경별 점검 (500 iteration 마다; 성공률 < 90% 또는 밀기 구간 비대칭 > 0.15 rad 면 중단)
uv run python scripts/monitor_getup.py logs/rsl_rl/microban_getup/<run> --task-id Microban-GetUp-5 \
    --train-pid <pid> --first 2500 --last 3999 --stop-asym 999 --stop-push-asym 0.15 \
    --min-success 0.9 --stall-below-success 0.9
# 체크포인트 하나 점검 / 밀기 대칭·착지 충격 측정
uv run python scripts/monitor_getup.py --task-id Microban-GetUp-5 --checkpoint-file <model.pt>
uv run python scripts/measure_push.py <model.pt>

# 일어나기만 재생 (viser) → http://localhost:8080
uv run microban-play Microban-GetUp-5 --viewer viser \
    --checkpoint-file logs/rsl_rl/microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt

# 일어나기 → 보행 전환 성공률 / viser 재생
uv run python scripts/getup_walk.py --num-envs 32 \
    --checkpoint logs/rsl_rl/microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt
uv run python scripts/getup_walk.py --num-envs 4 --viewer viser \
    --checkpoint logs/rsl_rl/microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt

# 영상 저장 (20 초, 측면, 1280x720, 50 fps)
uv run python scripts/record_getup_walk.py \
    --checkpoint logs/rsl_rl/microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt \
    --output ~/mero/videos/getup_walk.mp4
```

### 누운(supine) 상태에서 일어나기 — 보류

엎드린 버전의 고관절 범위 문제 때문에 누운 상태에서 일어나기를 검토했습니다 (별도 세션에서 수행). 만든 것:
`src/microban_rl/supine.py` (누운 시작 자세: 정착 높이 0.031 m, 등 88% · 뒤꿈치 12% 하중, 유지 토크 ≤ 0.05 Nm,
랜덤화, 앞으로 구름 종료 조건 `belly_down`), `keyframe_editor/init/getup_back_init.lz4` (supine 0 초 → crouch 5 초 →
stand_target 6.5 초, `scripts/make_getup_back_init.py`), 분석 스크립트 `scripts/analysis/supine/`.

가능성 점검 결과 (CPU MuJoCo, 체중 W = 8.0 N, 토크 한계 0.64 Nm) — **정적으로 이어 갈 수 없는 구간이 두 곳**:

| 구간 | 결과 |
| --- | --- |
| 누움 | 안정 (토크 ≤ 0.05 Nm) |
| 팔 뒤로 짚기 + 몸통 0 → 55° | 정적 불가: 엉덩이 접점이 무게중심보다 앞이라 뒤로 넘어감. 팔이 짧아(어깨→손 11.5 cm) 손을 무게중심 뒤에 둘 수 있는 건 몸통 35° 이상. 고관절만 쓰면 다리가 들려 V 자가 됨 → 반동 필요 |
| 앉기 60 ~ 80° | 손 없이도 안정 (토크 ≤ 0.05 Nm) |
| 무릎 굽혀 발을 엉덩이 근처로 | 고관절 ±90° 가 막음: 발 중심이 고관절 앞 12 ~ 14 cm 보다 가까이 오지 못함 |
| 손으로 밀어 엉덩이를 발 위로 | 정적 불가: 손을 댄 채 엉덩이는 2 ~ 4 cm 만 이동, 그때도 무게중심이 뒤꿈치보다 11 ~ 13 cm 뒤. 손을 뗄 때 무게중심 속도 ≥ 1.0 ~ 1.1 m/s (충격량 0.8 ~ 0.9 N·s) 필요 — 손 힘(마찰·어깨 토크 한계)만으로는 부족, 발 끌기·고관절 펴기·몸통 반동이 함께 필요 |
| crouch / stand_target | crouch 안정 (무릎 0.12 Nm), stand_target 은 정책 균형 필요 (엎드린 버전과 같음) |

보류 이유: 엎드린 경로는 손·발이 늘 몸을 받치며 이어 갈 수 있는 반면 (손 밀기 한 번만 동적), 누운 경로는 반동이
두 번 필요해 난이도가 높습니다. 엎드린 버전을 먼저 완성했고, 누운 상태는 "옆으로 굴러 엎드린 뒤 엎드린 일어나기로
잇기"가 더 쉬운 대안으로 제안되어 있습니다 (구르기 동작은 아직 계산하지 않음).

## 보행 (최종: `Microban-Walk-DR`, walk.onnx 초기화 + 미세조정)

실물 MERO 로봇(`mero_walk/` 런타임)에서 쓸 보행 정책입니다. 최종: **`checkpoints/walk_dr_model_1000.pt`**
(원본 `logs/rsl_rl/microban_walk/2026-11-20_06-29-45_dr/model_1000.pt`), ONNX **`mero_walk/models/mero_walk_dr.onnx`**.
학습은 원격 PC (RTX 4070 Ti SUPER, 4096 env, 약 1 s/iteration) 에서 했습니다.

### 환경 (`walk_env_cfg.py`)

- **관측·행동 = walk.onnx 형식** (`mero_walk/mero/policy.py` 가 그대로 읽음): 몸통 각속도 (3, 몸통 프레임 → `--gyro-frame body`),
  투영 중력 (3), 관절각 − 기본 자세 (18, head 제외), 관절 속도 (18), 직전 행동 (18), 명령 vx·vy·wz (3) = 63 → 18.
  목표 = 기본 자세 + 1.0 × 행동, head 는 0 유지.
- **실물 조건**: 제어 30 Hz (115200 bps 버스, 물리 1/180 s × 6), 5 V 공급에서 `kp_walk` 190 ≈ 0.275 Nm/rad, 토크 한계 0.64 Nm,
  명령 지연 2–5 물리 스텝 (11–28 ms), 자이로·중력 관측 지연 0–2 스텝, 관절 속도 지연 0–1 스텝.
- 보상·이벤트·PPO 는 walk.onnx 를 만든 Rhoban 학습 레포 [MarcDcls/mjlab_microban](https://github.com/MarcDcls/mjlab_microban)
  (Apache-2.0, commit `d594a60`) 의 값을 따름: 속도 추종, 10° 앞기울기 upright, 자세, air time 0.125–0.3 s, 발 높이 2 cm,
  발 미끄럼, 발 간격 ≥ 8 cm, 자기 충돌, 1–3 초마다 ±0.5 m/s 밀기, 관절 armature·마찰 ±10 %, 몸통 무게중심 ±5 mm.
- 명령 범위: vx −0.3–0.4, vy ±0.2 m/s, wz ±0.75 rad/s, 10 % 정지 명령.
- **`Microban-Walk-DR` 추가 랜덤화** (env 마다 리셋 때):
  - 바닥 (`floor_dr.py`): 마찰 0.4–1.2, 접촉 solref timeconst 0.012–0.04 s · dampratio 0.7–1.3 (딱딱한 바닥 ~ 매트).
    로봇 충돌 geom 에 priority 1 을 줘서 이 값이 로봇–바닥 접촉을 그대로 결정.
  - 기어 백래시 (`backlash.py`): 관절마다 유격 0–3°. 모터(엔코더) 쪽 각은 목표를 따라가다 유격 끝에서 링크를 밀고,
    유격 안에서는 링크에 토크가 전달되지 않음. **actor 는 엔코더 각을 관측** (critic 은 실제 링크 각).
  - 보정 오차 ±2° (엔코더가 실제보다 c 만큼 다르게 읽고, 목표도 그만큼 어긋남), kp × 0.75–1.3, 토크 한계 × 0.75–1.0.
- `Microban-Walk` 는 같은 환경에서 백래시·보정 오차·바닥 DR 을 뺀 것 (발 마찰 0.3–1.2 만, kp × 0.9–1.1).

### 학습 과정

| 시도 | 내용 | 결과 |
| --- | --- | --- |
| run 1–4 | 직접 만든 보상 (좁은 추종 std 0.1, 행동 std 0.25 rad), 이후 보행 위상 클럭 + 접지 스케줄 보상, 약한 DR | 모두 **제자리에 서 있기**로 수렴 (결정적 정책 vx = 0) |
| upstream 레시피 | mjlab_microban 설정 그대로 (행동 scale 1, 초기 std 1, entropy 0.01) | 700 iteration 까지 여전히 서 있음 (upstream 은 15,000 iteration 학습) |
| **walk.onnx 초기화** | walk.onnx 는 우리 actor 와 구조가 같음 (63→512→256→128→18 ELU, 관측 정규화). IMU 프레임 자이로 → 몸통 프레임 변환은 부호 있는 축 치환이라 정규화·첫 층에 접어 넣음 (`scripts/init_from_walk_onnx.py`, 출력 차 3.8e−6) | iteration 0 부터 30 Hz·전체 DR 에서 걸음 |
| **미세조정** | `Microban-Walk-DR`, 초기 행동 std 0.15, entropy 0.002, 2000 iteration (약 33 분) | model_1000 선택 (아래) |

### 결과 (`scripts/export_walk_onnx.py`, `Microban-Walk-DR` 재생 설정, 명령마다 64 env × 10 초)

넘어짐 비율 (8 개 명령 평균). 기본 = 밀기 없음, DR 은 학습 분포 그대로. 최악 = 백래시 3°, kp·토크 × 0.75–0.8, 바닥 마찰 0.4–0.5.

| 정책 | 기본 | 밀기 (±0.5 m/s, 1–3 초마다) | 최악 DR | 밀기 + 최악 DR |
| --- | ---: | ---: | ---: | ---: |
| walk.onnx (초기화 그대로) | 0 % | 7.8 % | 0.2 % | 31 % |
| **미세조정 model_1000 (최종)** | **0 %** | **2.5 %** | **0 %** | 29 % |
| 미세조정 model_1999 | 0 % | 2.0 % | 1.8 % | 30 % |

model_1000 속도 추종 (기본 조건, 뒤 2/3 평균):

| 명령 (vx, vy, wz) | 측정 vx, vy, wz | walk.onnx 초기화 |
| --- | --- | --- |
| (0, 0, 0) | +0.006, +0.002, −0.005 | 0, 0, 0 |
| (0.1, 0, 0) | +0.086, −0.005, +0.013 | +0.087 |
| (0.2, 0, 0) | +0.151, +0.003, −0.007 | +0.171 |
| (0.3, 0, 0) | +0.209, −0.006, −0.003 | +0.246 |
| (−0.15, 0, 0) | −0.113, +0.004, −0.001 | −0.144 |
| (0, 0.1, 0) | +0.007, +0.036, +0.002 | vy +0.038 |
| (0, 0, 0.5) | +0.010, −0.007, +0.366 | wz +0.453 |
| (0.15, 0, −0.4) | +0.119, +0.001, −0.264 | +0.127, wz −0.272 |

- 미세조정으로 밀기에 넘어지는 비율이 7.8 → 2.5 % 로 줄었고, 최악 DR 에서도 넘어지지 않습니다.
- 대신 빠른 명령을 덜 따라갑니다 (0.3 m/s 명령에 0.21 m/s, walk.onnx 0.25). 실물에서는 명령을 조금 크게 주면 됩니다.
- 밀기와 최악 DR 이 겹치면 둘 다 30 % 정도 넘어집니다 (가장 약한 모터·미끄러운 바닥에 강한 밀기).
- 실물 전 확인할 것: 30 Hz 실제 주기·지연 (`mero bench`), 5 V 에서의 실제 kp (190 레지스터 ≈ 0.275 Nm/rad 가정), 관절 유격 실측.

### 실물 결과와 저전류 미세조정 (`Microban-Walk-DR2`, 2026-11-20)

실물 걷기 로그 (`mero_walk/logs/`, 런 구분은 `mero_walk/logs/README.md`) 분석:

| 증상 | 로그 근거 | 원인 |
| --- | --- | --- |
| walk.onnx 가 피치 10° → 20° 출렁인 뒤 뒤로 넘어짐 | 자이로 수정 전 런 6개 | IMU 칩 장착이 업스트림과 달라 walk.onnx 입력 자이로 x·y 부호가 뒤집힘 (기대값과 상관 −0.99/−0.98). 런타임에서 수정됨 (`429fc1d`) |
| 수정 후 walk.onnx 가 서 있을 때 앞으로 15–20° (시뮬 12°) | `081758`, `082346`, `082003` | 발목 피치가 목표보다 −7 ~ −12° 처짐 (오른쪽이 더) |
| 걸으면 오른쪽으로 22–27° 기욺, kp_walk 220 에서 오른발로 체중이 넘어가며 넘어짐 | `081935`, `081954`, `082003` | 오른쪽 발목 롤·고관절 롤이 똑바로 서 있을 때도 왼쪽보다 3–15° 더 처짐 (q − 목표는 엔코더 기준이라 원점과 무관한 힘 부족) |
| `mero_walk_dr.onnx` 1초 만에 오른쪽 44° | `081938` | 관측은 학습 분포 안 (첫 10 스텝 \|z\| 0.2–0.4). 속도 0 에서 제자리걸음 → 오른쪽 무릎이 60–90° 꺾인 채 안 펴짐 |
| 걷다가·서 있다가 전원 끊김, 재부팅 | 여러 런 (`.json` 빈 파일, `event` 없이 끝남) | 모터 전압 4.58–4.87 V (1초에 모터 하나씩 읽은 평균이라 순간 강하는 안 보임). 시뮬 추정 걷기 전류는 모터만 평균 3–4.5 A, 순간 5–6 A 이상 → 레귤레이터 용량 부족이 유력 |

정리하면 **실물 모터가 시뮬보다 약하고 (5 V 스톨 토크 0.52 Nm vs 시뮬 한계 0.64 Nm, 부하 때 전압 강하), 오른쪽이 더 약하며, 걷는 전류가 레귤레이터에 부담**입니다.
소프트웨어로 `Microban-Walk-DR2` 를 DR model_1000 에서 1000 iteration 미세조정했습니다 (원격 2048 env, 약 35 분):
- `no_stepping` −1: 명령이 0 이면 발을 떼지 않기 (mjlab_microban 항목)
- `motor_current` −0.1/A: Σ|τ|/kt (XL330 kt 0.366 Nm/A)
- 토크 한계 × 0.6–0.95 (DR 은 0.75–1.0), 정지 명령 20 %

같은 조건 (`Microban-Walk-DR2` 재생, 명령마다 64 env × 10 초; 전류는 32 env × 8 초):

| 정책 | 서 있기 전류 | 걷기 전류 vx 0.1–0.3 (p99) | 기본 | 밀기 | 최악 DR | 밀기 + 최악 | vx 0.1 / 0.2 / 0.3 명령 → 실제 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DR model_1000 | 3.47 A | 3.66–4.51 A (6.3) | 0 % | 5.1 % | 0.2 % | 30 % | 0.08 / 0.15 / 0.21 |
| **DR2 model_1400 (최종)** | **0.83 A** | **2.90–3.58 A (5.4)** | **0 %** | **3.5 %** | **0 %** | **17 %** | 0.07 / 0.12 / 0.17 |
| DR2 model_1999 | 0.87 A | 2.75–3.55 A (5.7) | 0 % | 3.5 % | 0.2 % | 22 % | 0.07 / 0.12 / 0.16 |

- 최종: **`checkpoints/walk_dr2_model_1400.pt`** (원본 `logs/rsl_rl/microban_walk/2026-11-20_10-34-35_dr2/model_1400.pt`), ONNX **`mero_walk/models/mero_walk_dr2.onnx`** (`config.5v.example.json` 기본 정책).
- 런타임 시뮬 (MuJoCo 플랜트, MERO IMU 장착, 30 Hz, 4.9 V, kp 620/190) 에서 vx 0–0.3 모두 10 초 넘어짐 없음. vx 0.3 에서는 전진이 오히려 줄어 (0.77 m / 10 s) 처음엔 vx ≤ 0.2 권장. 뒤로 걷기는 거의 안 됨 (−0.15 → −0.04 m/s).
- **UBEC 제거 (2026-11-20 오후): 모터를 2S 배터리에 직결** → `mero_walk/config.2s.example.json` (kp 400/125, 30 Hz, 정책 `mero_walk_dr2.onnx`).
  런타임 시뮬 7.6 V·kp 400/125 에서 walk.onnx·DR·DR2 모두 vx 0–0.3 10 초 넘어짐 없음.
  배터리 직결용으로 토크 한계를 ×0.8–1.0 으로 되돌린 `Microban-Walk-DR3` (DR2 model_1400 에서 600 it) 도 시험했지만, 같은 조건에서 속도·전류는 같고 약한 모터 (최악 DR) 넘어짐만 늘어 (model_1700 1.2 %, model_1999 11.5 % vs DR2 0 %) 채택하지 않음.
- 하드웨어 쪽 확인 (5 V 구성 때): UBEC 정격 전류 (8–10 A 이상 권장, 또는 오렌지파이 전원 분리), 오른쪽 발목 롤·고관절 롤 모터 (로봇을 들어 올려 무부하로 같은 자세를 줬을 때도 오른쪽만 처지면 모터·마찰·혼 문제, 하중에서만 처지면 힘 부족).

### 실행

```bash
cd ~/mero/microban_rl

# 1) walk.onnx 로 초기화 체크포인트 만들기 (--template: 같은 태스크의 아무 체크포인트, 예: Microban-Walk 를 몇 iteration 돌린 model_0.pt)
uv run microban-train Microban-Walk --agent.max-iterations 1
uv run python scripts/init_from_walk_onnx.py --template logs/rsl_rl/microban_walk/<run>/model_0.pt \
    --out logs/rsl_rl/microban_walk/walkonnx_init/model_0.pt

# 2) 미세조정 (git 없는 PC 에서는 GIT_PYTHON_REFRESH=quiet)
uv run microban-train Microban-Walk-DR --agent.resume True --agent.load-run walkonnx_init --agent.load-checkpoint model_0.pt

# 3) 저전류 미세조정 (DR model_1000 에서)
uv run microban-train Microban-Walk-DR2 --agent.resume True --agent.load-run <dr run> --agent.load-checkpoint model_1000.pt

# 걷는 중 피치·모터 전류·관절 추종 (시뮬)
TASK=Microban-Walk-DR2 uv run python scripts/walk_pitch.py ../checkpoints/walk_dr2_model_1400.pt

# 평가 (밀기 / 최악 DR) + mero_walk 용 ONNX 내보내기
TASK=Microban-Walk-DR uv run python scripts/export_walk_onnx.py ../checkpoints/walk_dr_model_1000.pt --push --hard
TASK=Microban-Walk-DR uv run python scripts/export_walk_onnx.py ../checkpoints/walk_dr_model_1000.pt \
    --out ../mero_walk/models/mero_walk_dr.onnx

# 환경 점검 (관측 형식, 바닥 DR·백래시 적용 확인) / 재생
uv run python scripts/check_walk_env.py --task Microban-Walk-DR
uv run microban-play Microban-Walk-DR --viewer viser --checkpoint-file ../checkpoints/walk_dr_model_1000.pt

# 로봇에서 (mero_walk/README.md 5장)
mero run --policy models/mero_walk_dr.onnx --gyro-frame body
```

## ONNX 내보내기

최종 정책 3개 (Stand, Crawl-Arms-2, GetUp-5) 의 actor 를 관측 정규화 포함 ONNX 로 내보냈습니다 (`onnx/`).
메타데이터 (관측 순서·차원, 관절 순서, 기본 자세, action scale, 제어 주기, kp, 토크·속도 제한) 와 실물 사용 점검은
[`onnx/README.md`](onnx/README.md) 참조. 다시 만들기: `cd microban_rl && uv run python scripts/export_onnx.py`.

## 라이선스

이 레포는 **GPL-3.0** (`LICENSE`) 입니다. microban 의 제어 소프트웨어(로봇 MJCF 포함)가 GPL-3.0 이고, 아래 파일이
microban 에서 가져오거나 수정한 내용을 포함하므로 배포 시 GPL-3.0 을 따라야 합니다.

| 파일 | microban 에서 가져온 것 |
| --- | --- |
| `keyframe_editor/microban_scene.xml` | microban `src/model/mjcf/scene.xml`·`robot.xml` 을 수정해 저장한 MJCF (토크 한계, 시뮬레이션 옵션, 손 사이트, 키 자세) |
| `microban_rl/scripts/walk_microban_sim.py` | microban `src/imu_reader.py` 의 IMU 변환 함수 복사, `sim/mujoco_controller.py`·`moves/walk.py` 구조 재현 |
| `microban_rl/src/microban_rl/walk_onnx.py` | microban `walk.onnx` 의 인터페이스·상수 (관측 순서, IMU 장착 쿼터니언, KP_RL, 센서 지연) 재현, walk.onnx 가중치를 실행 시 읽음 |
| `microban_rl/src/microban_rl/walk_env_cfg.py`, `scripts/init_from_walk_onnx.py` | walk.onnx 가중치로 actor 초기화 (실행 시 서브모듈에서 읽음), 보상·DR 값은 mjlab_microban (Apache-2.0) 을 따름 |
| `microban_rl/src/microban_rl/robot.py` | MJCF 는 서브모듈에서 실행 시 읽음 (복사 안 함), 기본 자세·KP 상수는 microban `constants.py` / walk.onnx 메타데이터 값 |

mjlab (Apache-2.0) 과 robot_keyframe_kit (MIT) 은 서브모듈로 수정 없이 사용하며 GPL-3.0 과 호환됩니다.
microban 의 CAD 폴더(CC BY-NC-SA 4.0)는 사용하지 않았습니다.

## 문제 해결

- `nconmax overflow`: `stand_env_cfg.py` 의 `SimulationCfg(nconmax=..., njmax=...)` 를 `None`(자동) 또는 더 큰 값으로.
- GPU 메모리 부족: `--env.scene.num-envs 2048` 처럼 env 수를 줄이세요.
- 기본 로거는 tensorboard 입니다. W&B 를 쓰려면 `--agent.logger wandb` (먼저 `uv run wandb login`).
- 첫 실행 시 Warp 커널 컴파일로 수십 초 걸릴 수 있습니다 (캐시: `~/.cache/warp`).
