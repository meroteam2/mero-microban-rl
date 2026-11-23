# 최종 체크포인트

학습 로그 폴더(`microban_rl/logs/`)는 레포에 올리지 않았고, 최종 정책만 복사해 두었습니다.
모두 rsl-rl 5.5.1 / MJLab 1.6.0 형식 (`model_*.pt`: actor·critic·옵티마이저 상태).

| 파일 | 태스크 | 원본 (`microban_rl/logs/rsl_rl/...`) | 결과 |
| --- | --- | --- | --- |
| `stand_model_299.pt` | `Microban-Stand` | `microban_stand/2026-10-30_11-37-20/model_299.pt` | 300 iteration, 넘어짐 0 |
| `crawl_arms2_model_5497.pt` | `Microban-Crawl-Arms-2` | `microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt` | 0.191 m/s, 비대칭 0.002 rad |
| `getup5_model_2500.pt` | `Microban-GetUp-5` | `microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt` | 일어나기 성공 100%, walk.onnx 전환 100% |
| `walk_dr2_model_1400.pt` | `Microban-Walk-DR2` | `microban_walk/2026-11-20_10-34-35_dr2/model_1400.pt` | DR model_1000 → 저전류 미세조정 (서 있기 0.83 A, 밀기 넘어짐 3.5%). ONNX: `mero_walk/models/mero_walk_dr2.onnx` |
| `walk_dr4_model_2000.pt` | `Microban-Walk-DR4` | `microban_walk/2026-11-23_12-38-28_dr4/model_2000.pt` | DR2 model_1400 → 관절 원점 ±3°·IMU 기울기 ±3°·무게중심 ±10 mm (넓힌 DR 에서 밀기 넘어짐 11.6 → 5.3%). ONNX: `mero_walk/models/mero_walk_dr4.onnx` (실물 권장) |
| `walk_dr_model_1000.pt` | `Microban-Walk-DR` | `microban_walk/2026-11-20_06-29-45_dr/model_1000.pt` | walk.onnx 초기화 + 바닥 DR·백래시 미세조정, 밀기 넘어짐 2.5% (ONNX: `mero_walk/models/mero_walk_dr.onnx`) |

재생 예 (`microban_rl/` 에서):

```bash
uv run microban-play Microban-Stand --viewer viser --checkpoint-file ../checkpoints/stand_model_299.pt
uv run microban-play Microban-Crawl-Arms-2 --viewer viser --checkpoint-file ../checkpoints/crawl_arms2_model_5497.pt
uv run python scripts/getup_walk.py --num-envs 4 --viewer viser --checkpoint ../checkpoints/getup5_model_2500.pt
uv run microban-play Microban-Walk-DR --viewer viser --checkpoint-file ../checkpoints/walk_dr_model_1000.pt
```

ONNX 로 내보낸 actor 는 `../onnx/` 에 있습니다.
