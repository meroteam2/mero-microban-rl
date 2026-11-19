"""Calibration workflow on a plant whose 'physical' build differs from the defaults."""
import io
import math
import numpy as np
import mujoco
from mero import config, term
from mero.app_calib import CalibApp, ZERO, snap_zero
from mero.bus import MotorBus
from mero.calibration import Calibration, JointCal, quat_to_mat, mat_to_quat, UPSTREAM_MOUNT, AXIS_ROTATIONS, IMU_POSES
from mero.dxl import Protocol2
from mero.imu import ImuFrames, ImuReader, quat_from_two_vectors
from mero.joints import JOINTS, DEFAULT_SIGN
from mero.sim.plant import Plant, FakeImuDevice


def _true_cal():
    t = Calibration()
    t.joints["left_knee"].sign *= -1                      # motor mounted mirrored
    t.joints["right_hip_roll"].zero = math.radians(90)     # horn a quarter turn off
    t.joints["left_elbow"].zero = math.radians(-90)
    t.joints["head"].zero = math.radians(4)                # small leftover: not snapped away
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
    assert abs(math.degrees(app.q["right_hip_roll"])) > 85
    app.handle_key("z"); assert app.mode == ZERO
    app.handle_key("A"); app.handle_key("y")
    app.update()
    assert abs(math.degrees(app.cal.joints["right_hip_roll"].zero) - 90) < 1e-6
    assert abs(math.degrees(app.cal.joints["left_elbow"].zero) + 90) < 1e-6
    assert app.cal.joints["head"].zero == 0.0             # 4° snaps to 0°, error stays visible
    assert abs(abs(math.degrees(app.q["head"])) - 4) < 0.2
    assert max(abs(v) for n, v in app.q.items() if n != "head") < math.radians(0.2)
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
        if n != "head":
            assert abs(app.q[n] - pose.get(n, 0.0)) < 0.01, n
    app.render(force=True)
    app.handle_key("s")
    re = Calibration.load(tmp_path / "calibration.json")
    for n in JOINTS:
        assert re.joints[n].sign == app.cal.joints[n].sign
        assert abs(re.joints[n].zero - app.cal.joints[n].zero) < 1e-4
    assert re.joints["left_knee"].sign == -DEFAULT_SIGN["left_knee"]


def test_snap_zero_nearest_quarter_turn():
    for deg, want, left in [(0, 0, 0), (3, 0, 3), (-44, 0, -44), (46, 90, -44), (93, 90, 3),
                            (-87, -90, 3), (178, -180, -2), (-179, -180, 1), (-135.5, -180, 44.5)]:
        z, r = snap_zero(math.radians(deg))
        assert abs(math.degrees(z) - want) < 1e-9, deg
        assert abs(math.degrees(r) - left) < 1e-9, deg


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


def test_exact_zero_trims_small_leftover(tmp_path):
    app, plant, _ = _app(tmp_path)
    _pose(plant, {})
    app.update()
    app.handle_key("A"); app.handle_key("y"); app.update()
    assert abs(abs(math.degrees(app.q["head"])) - 4) < 0.2       # 90° snap leaves the 4°
    app.sel = JOINTS.index("head")
    app.handle_key("x"); app.update()
    assert abs(math.degrees(app.q["head"])) < 0.2                # exact zero removes it
    assert abs(abs(math.degrees(app.cal.joints["head"].zero)) - 4) < 0.2
    assert abs(math.degrees(app.cal.joints["right_hip_roll"].zero) - 90) < 1e-6   # others untouched


def test_stand_hold_and_live_trim(tmp_path):
    from mero.app_calib import HOLD
    from mero.joints import STAND_POSE
    app, plant, _ = _app(tmp_path)
    _pose(plant, {})
    app.update()
    app.handle_key("A"); app.handle_key("y"); app.update()     # head keeps its 4° (snap)
    app.cfg["control"]["ramp_s"] = 0.3
    app.handle_key("t")
    assert app.mode == HOLD and all(m.torque for m in plant.motors.values())

    def settle(sec):
        for _ in range(int(sec / 0.02)):
            app.update(); plant.step(0.02)
    settle(1.5)
    assert not app.ramping
    k = plant.qadr["head"]
    err = math.degrees(plant.d.qpos[k] - STAND_POSE["head"])
    assert abs(abs(err) - 4) < 1.0, err                         # the real head is 4° off the stand pose
    app.handle_key("f")                                         # other keys are refused while holding
    assert app.cal.joints["head"].sign == DEFAULT_SIGN["head"]
    app.sel = JOINTS.index("head")
    for _ in range(4):
        app.handle_key(term.KEY_LEFT if err > 0 else term.KEY_RIGHT)   # → moves the real joint +
    settle(1.0)
    assert abs(math.degrees(plant.d.qpos[k] - STAND_POSE["head"])) < 1.0
    assert app.dirty
    app.handle_key("s")
    assert abs(Calibration.load(tmp_path / "calibration.json").joints["head"].zero - app.cal.joints["head"].zero) < 1e-4
    app.handle_key("t")
    assert app.mode != HOLD and not any(m.torque for m in plant.motors.values())
