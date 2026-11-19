import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import gz_world
from mero.calibration import quat_to_mat, AXIS_ROTATIONS, mat_to_quat


def _rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def test_rpy_roundtrip_all_visuals_and_axis_rotations():
    quats = [v["quat"] for vis in gz_world.parse_bodies().values() for v in vis]
    quats += [mat_to_quat(R) for R in AXIS_ROTATIONS]
    rng = np.random.default_rng(0)
    quats += [tuple(q / np.linalg.norm(q)) for q in rng.normal(size=(200, 4))]
    for q in quats:
        assert np.allclose(quat_to_mat(q), _rpy_mat(*gz_world._quat_to_rpy(q)), atol=1e-6), q


def test_world_has_live_and_ghost_for_every_body(tmp_path):
    names = gz_world.build_world(tmp_path / "w.sdf")
    txt = (tmp_path / "w.sdf").read_text()
    assert len(names) == 20
    for n in names:
        assert f'name="live_{n}"' in txt and f'name="ghost_{n}"' in txt
