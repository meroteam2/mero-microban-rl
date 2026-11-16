# ONNX 정책

최종 정책 3개의 actor 를 ONNX 로 내보낸 것입니다 (`microban_rl/scripts/export_onnx.py`).
관측 정규화(평균·표준편차)가 그래프 안에 포함되어 있으므로 **정규화하지 않은 관측을 그대로** 넣으면 됩니다.
출력은 Gaussian 정책의 평균(결정적 행동)입니다.

| 파일 | 태스크 / 체크포인트 | 입력 `obs` | 출력 `actions` | torch vs onnxruntime (기록된 관측 240개) |
| --- | --- | --- | --- | --- |
| `microban_stand.onnx` | `Microban-Stand` model_299 | [1, 63] | [1, 19] | 최대 차 3.6e-7 |
| `microban_crawl_arms2.onnx` | `Microban-Crawl-Arms-2` model_5497 | [1, 65] | [1, 19] | 최대 차 1.9e-6 |
| `microban_getup5.onnx` | `Microban-GetUp-5` model_2500 | [1, 83] | [1, 19] | 최대 차 7.5e-7 |
| `getup_reference.npz` | 일어나기 참조 궤적 (`microban_getup5.onnx` 에 필요) | | | |

배치 크기는 1 로 고정입니다.

## 메타데이터

walk.onnx 처럼 `metadata_props` 에 문자열로 들어 있습니다.

```python
import onnx, onnxruntime as ort
m = onnx.load("microban_getup5.onnx")
meta = {p.key: p.value for p in m.metadata_props}
sess = ort.InferenceSession("microban_getup5.onnx")
action = sess.run(None, {"obs": obs[None].astype("float32")})[0][0]
```

| 키 | 내용 |
| --- | --- |
| `joint_names` | 관절 순서 (행동·`joint_pos`·`joint_vel` 공통): head, right_shoulder_pitch, right_shoulder_roll, right_elbow, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll, left_shoulder_pitch, …, left_ankle_roll (MJCF 순서, head 포함 19개) |
| `observation_names`, `observation_dims`, `observation_scales` | 관측 항목 순서·차원·스케일 (스케일은 모두 1) |
| `observation_description` | 항목별 정의 (JSON) |
| `default_joint_pos` | 기본 자세 (walk.onnx 의 기본 자세와 같음) |
| `action_scale`, `action_rule` | 출력 → 관절 목표 변환식 |
| `control_dt` | 0.02 s (50 Hz) |
| `kp`, `kd`, `torque_limit`, `position_actuator` | 0.88 Nm/rad (= XL330 KP 400 레지스터), 0.01 Nm·s/rad, 0.6405 Nm (1.75 A × 0.366), τ = kp(target − q) − kd·q̇ |
| `target_speed_limit` | 목표 관절각 변화율 한계 rad/s (Crawl-Arms-2, GetUp-5: 10.304 = XL330 무부하 속도의 80%; Stand: none) |
| (GetUp) `reference_file`, `reference_t_start`, `reference_t_end`, `push_window`, `push_clip_joints`, `push_clip_rad` | 참조 궤적과 밀기 구간 제한 |

## 관측과 출력 (정책별)

공통 항목 (모두 몸통 기준, 단위 rad, rad/s):

| 항목 | 차원 | 정의 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | 몸통 프레임 각속도. 실물: IMU 자이로를 IMU 장착 쿼터니언 (0.5, −0.5, −0.5, 0.5) 으로 몸통 프레임으로 회전 (walk.onnx 는 IMU 프레임 그대로 쓰므로 다름) |
| `projected_gravity` | 3 | 몸통 프레임에서 본 중력 방향 단위벡터 (똑바로 서면 (0, 0, −1)) |
| `joint_pos` | 19 | q − default_joint_pos |
| `joint_vel` | 19 | q̇ |
| `actions` | 19 | 직전 스텝 정책 출력 (스케일·클리핑 전, 시작 시 0) |

| 정책 | 관측 (순서대로) | 관절 목표 |
| --- | --- | --- |
| Stand | base_ang_vel, projected_gravity, joint_pos, joint_vel, actions (63) | default + 0.25 × action |
| Crawl-Arms-2 | base_ang_vel, projected_gravity, **heading (2)**, joint_pos, joint_vel, actions (65) | default + 0.5 × action, 그다음 직전 목표 대비 ±10.304 × 0.02 rad 로 제한 (첫 스텝은 측정 관절각 기준) |
| GetUp-5 | base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, **getup_phase (1)**, **ref_joint_pos (19)** (83) | 아래 참조 |

`heading` = (sin, cos) of (머리 방향(몸통 +z)을 바닥에 투영한 방향 − 시작 시 진행 방향). 엎드린 자세에서 머리 쪽이 진행 방향입니다.
실물에서는 시작 시 IMU yaw 를 기준으로 한 상대 yaw 를 쓰면 됩니다.

## 일어나기 정책 (residual) 을 로봇에서 쓰는 법

`getup_reference.npz`:
- `q_ref` [621, 19]: 0.02 s 간격 참조 관절 목표 (0 ~ 12.4 s, `joint_names` 순서 = 메타데이터와 같음)
- `t`, `t_end` (12.4), `kf_names` / `kf_t` (키프레임 이름·시각)
- 나머지 (`tilt_ref`, `z_ref`, `replay_*`, `kf_root`, `kf_joint`) 는 학습용 (보상, 시작 상태) 으로 실물 실행에는 필요 없음

시작: 로봇이 엎드린 상태에서 k = 0. 매 스텝 (50 Hz):

```
t      = -0.5 + 0.02 * k                 # 처음 0.5 초는 엎드린 자세 유지 (참조 첫 행)
t_next = t + 0.02
row(x) = clip(round(x / 0.02), 0, 620)
obs = [ gyro_trunk (3), gravity_trunk (3), q - default (19), qdot (19), last_action (19),
        clip(t / 12.4, 0, 1)  (1),                       # getup_phase
        q_ref[row(t_next)] - default (19) ]               # ref_joint_pos (다음 스텝의 참조)
a = onnx(obs)                                            # last_action ← a (클리핑 전 값)
residual = 0.5 * a
if 8.0 <= t_next <= 10.4:                                # 밀기 구간
    residual[hip_yaw, hip_roll, ankle_roll (좌우)] = clip(…, -0.1, +0.1)
target = q_ref[row(t_next)] + residual
target = clip(target, prev_target - 0.206, prev_target + 0.206)   # 10.304 rad/s × 0.02 s; 첫 스텝 prev = 측정 관절각
모터 목표 ← target   (kp 0.88, 토크 한계 0.64 Nm)
```

- t > 12.4 이후 참조는 마지막 행 (stand_target = 기본 자세) 에 머물고, 정책은 그 주변에서 균형을 잡습니다 (학습 시 3 초).
- walk.onnx 로 넘길 때 (`scripts/getup_walk.py`): t = 13.4 까지 일어나기 → 13.4 ~ 14.4 초 일어나기 정책이 계속 균형 유지
  (참조 = 기본 자세) → 모터 KP 를 125 레지스터 (0.275 Nm/rad) 로 바꾸고 walk.onnx 시작. 고정 목표만으로 정지하면
  kp 0.88 로는 기본 자세를 못 버티고 뒤로 넘어집니다.

## 실물 사용 가능 여부 점검

### actor 관측 — 실물에서 얻을 수 있는가

| 관측 | Stand | Crawl-Arms-2 | GetUp-5 | 실물에서 | 판정 |
| --- | :---: | :---: | :---: | --- | --- |
| base_ang_vel | ✓ | ✓ | ✓ | IMU 자이로 (장착 방향 회전 필요) | 가능 |
| projected_gravity | ✓ | ✓ | ✓ | IMU 자세 추정 (AHRS) | 가능 (추정 지연 있음) |
| heading | | ✓ | | IMU yaw (시작 기준 상대값) | 가능 (자력계 없으면 자이로 적분 드리프트) |
| joint_pos | ✓ | ✓ | ✓ | Dynamixel present position | 가능 |
| joint_vel | ✓ | ✓ | ✓ | Dynamixel present velocity (0.229 rpm 단위, 잡음 큼) | 가능 (학습 시 ±1 rad/s 균일 잡음) |
| actions | ✓ | ✓ | ✓ | 직전 출력 저장 | 가능 |
| getup_phase | | | ✓ | 시작 후 경과 시간 | 가능 |
| ref_joint_pos | | | ✓ | `getup_reference.npz` | 가능 |

몸통 선속도(`base_lin_vel`), 몸통 높이(`root_height`), 바닥 접촉, 절대 위치는 **critic 에만** 쓰였고 ONNX 로 내보낸 actor 에는
없습니다. 따라서 **관측 때문에 재학습할 필요는 없습니다.**

### 그래도 실물 전에 바꿔서 재학습하는 것이 좋은 점 (sim-to-real 차이)

| 항목 | 현재 학습 | 실물 / microban 시뮬레이터 | 권장 |
| --- | --- | --- | --- |
| 센서·구동 지연 | 없음 | microban sim: 속도 1, 자이로 3, 자세 4 tick 지연, 구동 2 substep (walk.onnx 는 이 조건으로 학습) | 관측 지연 버퍼 + 지연 랜덤화를 넣어 재학습 (`walk_onnx.py` 의 지연 구현 재사용) |
| 모터 모델 | 이상적 PD (kp 0.88, kd 0.01, 0.64 Nm 클리핑) | XL330 + BAM 모델 (전압 강하, 전류 제한, 마찰) | BAM 모터 모델로 검증, kp·감쇠·토크 한계 랜덤화 |
| 도메인 랜덤화 | Stand: 미는 힘 / Crawl·GetUp: 관측 잡음만 | 질량·무게중심·마찰 오차 | 마찰·질량·무게중심 랜덤화 추가 (특히 GetUp 의 손 밀기·짧은 점프는 마찰에 민감) |
| 자이로 프레임 | 몸통 프레임 | IMU 프레임 | 실물 코드에서 장착 쿼터니언으로 회전 (재학습 불필요) |
| 게인 전환 | 시뮬레이터에서 env 별 즉시 변경 | 펌웨어 KP 레지스터 쓰기 (통신 지연) | 전환 직전 정지 구간에서 쓰기, 실물에서 확인 |
| 착지 충격 (GetUp) | 한 발 최대 3.0 × 체중 (p90 3.5) | 보행 중 walk.onnx 최대 3.2 × 체중과 같은 수준 | 서보 보호가 필요하면 착지 충격 페널티 추가 |
| heading (Crawl) | 시뮬레이터의 정확한 yaw | 자이로 적분 드리프트 | 짧은 시간은 그대로 사용, 길게 쓰면 heading 관측 잡음·드리프트 추가 학습 |

정리: 세 정책 모두 실물에서 얻을 수 있는 값만 관측합니다. 다만 센서 지연과 모터 모델(BAM)이 학습에 들어 있지 않아서,
실물에 올리기 전에 **지연 + 모터 모델 + 도메인 랜덤화를 넣고 재학습(또는 이어서 학습)** 한 뒤 microban 시뮬레이터
(`walk_microban_sim.py` 와 같은 BAM 설정)에서 먼저 확인하는 것을 권장합니다.
