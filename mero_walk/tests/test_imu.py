"""IMU frame handling and mount solving, using the MuJoCo plant as ground truth."""
import math
import numpy as np
import mujoco
import pytest
from mero.calibration import Calibration, solve_mount, IMU_POSES, quat_to_mat, UPSTREAM_MOUNT, AXIS_ROTATIONS, mat_to_quat
from mero.imu import ImuFrames, ImuReader
from mero.sim.plant import Plant, FakeImuDevice


def _set_tilt(plant, axis, deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, float), math.radians(deg))
    plant.d.qpos[3:7] = q
    plant.d.qpos[2] = 0.4
    plant.d.qvel[:] = 0
    mujoco.mj_forward(plant.m, plant.d)


def _settle(reader, n=400):
    for _ in range(n):
        reader.step(0.005)
    return reader.latest()


@pytest.mark.parametrize("axis,deg,expect", [
    ((0, 1, 0), 30, (math.sin(math.radians(30)), 0, -math.cos(math.radians(30)))),   # nose down
    ((1, 0, 0), -25, (0, math.sin(math.radians(25)), -math.cos(math.radians(25)))),  # left side down
    ((0, 0, 1), 0, (0, 0, -1)),
])
def test_projected_gravity(axis, deg, expect):
    p = Plant()
    _set_tilt(p, axis, deg)
    dev = FakeImuDevice(p, static=True)
    r = ImuReader(dev, ImuFrames(UPSTREAM_MOUNT))
    s = _settle(r)
    assert np.allclose(s.proj_grav, expect, atol=0.02), s.proj_grav


def test_gyro_frames_match_mjcf_site():
    p = Plant()
    _set_tilt(p, (0, 0, 1), 0)
    p.d.qvel[3:6] = [0.0, 0.0, 1.0]   # yaw rate, world frame == trunk frame here
    mujoco.mj_forward(p.m, p.d)
    r = ImuReader(FakeImuDevice(p), ImuFrames(UPSTREAM_MOUNT))
    r.step(0.005)
    s = r.latest()
    site_gyro = p.d.sensordata[p.s_gyr:p.s_gyr + 3]
    assert np.allclose(s.gyro_imu, site_gyro, atol=1e-6)        # what walk.onnx sees (gyro_frame=imu)
    assert np.allclose(s.gyro_body, [0, 0, 1.0], atol=1e-6)     # trunk frame


def _capture(p, mount):
    dev = FakeImuDevice(p, true_mount=mount, static=True)
    out = {}
    for name, (g_body, _) in IMU_POSES.items():
        # rotate trunk so that gravity appears as g_body in the trunk frame
        R_wb = quat_to_mat(np.array(_quat_for_gravity(g_body)))
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R_wb.flatten())
        p.d.qpos[3:7] = q; p.d.qvel[:] = 0; mujoco.mj_forward(p.m, p.d)
        acc, _ = dev.read()
        out[name] = acc
    return out


def _quat_for_gravity(g_body):
    from mero.imu import quat_from_two_vectors
    # rotation world<-body such that R^T (0,0,-1) = g_body  <=>  R g_body = (0,0,-1)
    return quat_from_two_vectors(g_body, [0, 0, -1])


@pytest.mark.parametrize("k", [0, 5, 11, 17, 23])
def test_solve_mount_recovers_axis_mounts(k):
    p = Plant()
    true_R = quat_to_mat(UPSTREAM_MOUNT) @ AXIS_ROTATIONS[k]
    true_q = mat_to_quat(true_R)
    res = solve_mount(_capture(p, true_q))
    assert res["snapped"] and res["err_deg"] < 3 and not res["improper"]
    assert np.allclose(res["R"], true_R, atol=1e-6)


def test_solve_mount_flags_mirrored_axis():
    p = Plant()
    samples = _capture(p, UPSTREAM_MOUNT)
    mirrored = {k: v * np.array([1, -1, 1]) for k, v in samples.items()}
    res = solve_mount(mirrored)
    assert res["improper"]
