# 분석 스크립트 (일회성)

키프레임 설계·실패 원인 분석에 쓴 스크립트를 임시 폴더에서 옮겨 온 것입니다. 정리된 도구가 아니라
당시 분석을 재현하기 위한 기록이며, `~/mero` = `/home/mero/mero` 경로와 키프레임 에디터 scene
(`keyframe_editor/microban_scene.xml`) 을 가정합니다. 중간 결과(`.npy`)는 `/tmp/microban_analysis/` 에 씁니다.
대부분 CPU MuJoCo 로 동작합니다 (`cd ~/mero/microban_rl && uv run python scripts/analysis/<폴더>/<파일>.py ...`).

| 폴더 | 내용 |
| --- | --- |
| `prone_feasibility/` | 엎드린 일어나기의 정적 가능성: `static_torque.py` (키프레임별 유지 토크·무게중심), `feasibility.py` (무게중심을 발 위에 둔 채 손이 바닥에 닿는지 / 손을 짚은 채 무게중심이 발끝에 얼마나 가까이 갈 수 있는지, 필요한 밀기 속도), `feas2.py` (깊은 쪼그림 경로), `fold2squat.py`, `rollup.py` (발 위 무게중심을 유지한 일어서기 경로의 토크), `com_analysis.py` (getup_1111 의 pike_1 → crouch 실패 원인). 실행: `feasibility.py scripts/analysis/prone_feasibility` (인자 = 이 폴더) |
| `keyframe_poses/` | getup_v6 중간 자세 계산: `gen_v6_poses.py`, `gen_v6_poses2.py` (pike_push, squat_deep), `gen_fold.py` (fold). 인자로 `prone_feasibility` 폴더를 줌 |
| `symmetry/` | 대칭 점검: `asym_pairs.py` (구간별 좌우 관절 차), `equiv.py` (관측 미러 일치·정책 미러 등변성), `sat_twist.py` (토크 한계 비율·hip_yaw 차), `stage_trace.py` (단계 조건 시간 추적). 인자 = 체크포인트 |
| `supine/` | 누운 상태 일어나기 가능성 점검 (별도 세션에서 수행): `common.py`·`ik.py`·`ik2.py`·`support.py` (공통 함수), `supine_settle.py` (누운 자세 정착), `run1~7.py`, `feas.py` (앉기, 발 당기기, 손으로 밀기 구간의 정적 가능성·필요 충격량). `supine/` 폴더 안에서 실행 |
