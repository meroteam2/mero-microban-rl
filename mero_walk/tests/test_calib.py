"""Calibration workflow on a plant whose 'physical' build differs from the defaults."""
import io
import math
import numpy as np
import mujoco
from mero import config, term
from mero.app_calib import CalibApp, ZERO
from mero.bus import MotorBus
from mero.calibration import Calibration, JointCal, quat_to_mat, mat_to_quat, UPSTREAM_MOUNT, AXIS_ROTATIONS, IMU_POSES
from mero.dxl import Protocol2
from mero.imu import ImuFrames, ImuReader, quat_from_two_vectors
from mero.joints import JOINTS, DEFAULT_SIGN
from mero.sim.plant import Plant, FakeImuDevice


def _true_cal():
    t = Calibration()
    t.joints["left_knee"].sign *= -1                      # motor mounted mirrored
    t.joints["right_hip_roll"].zero = math.radians(30)     # horn one spline off
    t.joints["left_elbow"].zero = math.radians(-14.4)
    return t


def _app(tmp_path, true_mount=None):
    cfg = config.load("/nonexistent")
    cfg["battery"]["state_file"] = None
    plant = Plant(true_cal=_true_cal(), fixed_base=True)
    cal = Calibration(path=tmp_path / "calibration.json")
    bus = MotorBus(Protocol2(ser=plant.ser), cal)
    bus.setup("auto")
    frames = ImuFrames(cal.mount_quat)
    imu = ImuReader(FakeImuDevice(plant, true_mount=true_mount, static=True), frames)
    app = CalibApp(cfg, bus, imu, frames, cal, stream=None, out=io.StringIO())
    return app, plant, imu


def _pose(plant, q):
    for n in JOINTS:
        plant.d.qpos[plant.qadr[n]] = q.get(n, 0.0)
    plant.d.qvel[:] = 0
    mujoco.mj_forward(plant.m, plant.d)


def test_zero_all_and_sign_flip(tmp_path):
    app, plant, _ = _app(tmp_path)
    _pose(plant, {})                          # robot physically in the zero pose
    app.update()
    assert abs(math.degrees(app.q["right_hip_roll"])) > 25
    app.handle_key("z"); assert app.mode == ZERO
    app.handle_key("A"); app.handle_key("y")
    app.update()
    assert max(abs(v) for v in app.q.values()) < math.radians(0.2)
    # sign: bend the real left knee +0.5 rad -> reading is mirrored until flipped
    _pose(plant, {"left_knee": 0.5})
    app.update()
    assert app.q["left_knee"] < -0.45
    app.sel = JOINTS.index("left_knee")
    app.handle_key("f")
    app.update()
    assert abs(app.q["left_knee"] - 0.5) < 0.01
    # all joints now track the plant
    pose = {"right_hip_roll": 0.3, "left_elbow": -0.7, "right_knee": 1.0}
    _pose(plant, pose)
    app.update()
    for n in JOINTS:
        assert abs(app.q[n] - pose.get(n, 0.0)) < 0.01, n
    app.render(force=True)
    app.handle_key("s")
    re = Calibration.load(tmp_path / "calibration.json")
    for n in JOINTS:
        assert re.joints[n].sign == app.cal.joints[n].sign
        assert abs(re.joints[n].zero - app.cal.joints[n].zero) < 1e-4
    assert re.joints["left_knee"].sign == -DEFAULT_SIGN["left_knee"]


def test_jog_moves_only_selected_joint(tmp_path):
    app, plant, _ = _app(tmp_path)
    _pose(plant, {})
    app.update()
    app.sel = JOINTS.index("right_knee")
    app.handle_key("j")
    for _ in range(4):
        app.handle_key(term.KEY_RIGHT)
    for _ in range(50):
        plant.step(0.02)
    app.update()
    assert app.jog_target == math.radians(20)
    assert abs(plant.d.qpos[plant.qadr["right_knee"]] - math.radians(20)) < math.radians(4)
    assert [n for n in JOINTS if plant.motors[__import__("mero.joints").joints.MOTOR_ID[n]].torque] == ["right_knee"]
    app.handle_key("j")
    assert not any(m.torque for m in plant.motors.values())


def test_imu_wizard_fixes_wrong_mount(tmp_path):
    wrong = mat_to_quat(quat_to_mat(UPSTREAM_MOUNT) @ AXIS_ROTATIONS[7])
    app, plant, imu = _app(tmp_path, true_mount=wrong)
    plant.fixed_base = False
    imu.start()
    try:
        app.handle_key("m")
        for name, (g_body, _) in IMU_POSES.items():
            q = quat_from_two_vectors(g_body, [0, 0, -1])
            plant.d.qpos[3:7] = q; plant.d.qvel[:] = 0; mujoco.mj_forward(plant.m, plant.d)
            import time; time.sleep(0.3)
            app.handle_key(term.KEY_ENTER)
        assert app.pending == "imu_apply", app.msg
        app.handle_key("y")
        assert np.allclose(quat_to_mat(app.cal.mount_quat), quat_to_mat(wrong), atol=1e-6)
        plant.d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(plant.m, plant.d)
        import time; time.sleep(1.5)
        assert np.allclose(imu.latest().proj_grav, [0, 0, -1], atol=0.03)
    finally:
        imu.stop()
