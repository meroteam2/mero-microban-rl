"""MuJoCo stand-in for the real robot, wired to the fake Dynamixel chain and IMU.

Lets the whole runtime (serial protocol, calibration, IMU frames, policy, safety)
run closed-loop without hardware. `true_cal` describes how the *physical* robot is
built; give it errors to check that the calibration tools catch them.
"""
from __future__ import annotations

import struct
from pathlib import Path

import mujoco
import numpy as np

from .. import dxl
from ..calibration import Calibration, quat_to_mat
from ..joints import JOINTS, MOTOR_ID, STAND_POSE
from .fake_dxl import FakeMotor, FakeSerial

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "model" / "mjcf" / "scene.xml"

# XL330 voltage-controlled model (upstream constants.py PROXY_* / BAM m6)
KT, RES, ERR_GAIN, I_MAX = 0.366, 2.811, 0.0028773775, 1.75


class Plant:
    def __init__(self, true_cal: Calibration | None = None, control_hz: float = 50.0,
                 vin: float = 7.6, standing: bool = True, scene: Path = SCENE, fixed_base: bool = False,
                 act_delay_s: float = 0.0):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.m.dof_damping[6:] = 0.0          # back-EMF term below provides the damping
        self.true_cal = true_cal or Calibration()
        self.dt_ctrl = 1.0 / control_hz
        self.vin = vin
        self.qadr = {n: self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS}
        self.vadr = {n: self.m.jnt_dofadr[self.m.joint(n).id] for n in JOINTS}
        self.act = {n: self.m.actuator(n).id for n in JOINTS}
        self.s_acc = self.m.sensor("imu_accel").adr[0]
        self.s_gyr = self.m.sensor("imu_ang_vel").adr[0]
        self.motors = {MOTOR_ID[n]: FakeMotor(MOTOR_ID[n]) for n in JOINTS}
        self.ser = FakeSerial(list(self.motors.values()))
        self.ser.on_write = self._on_write
        self.ser.before_read = self._sync_present
        self.target = {n: 0.0 for n in JOINTS}
        self.sim_time = 0.0
        self.fixed_base = fixed_base
        self.act_delay_s = act_delay_s      # bus latency: new goals reach the motors this much later
        self._prev_goals = None
        self.reset(standing and not fixed_base)

    # -------------------------------------------------------------- state
    def reset(self, standing: bool = True) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2] = 0.165 if standing else 0.30
        for n in JOINTS:
            self.d.qpos[self.qadr[n]] = STAND_POSE[n]
        mujoco.mj_forward(self.m, self.d)
        for n in JOINTS:
            self.target[n] = STAND_POSE[n]
            mid = MOTOR_ID[n]
            raw = self.true_cal.joints[n].to_raw(STAND_POSE[n])
            struct.pack_into("<i", self.motors[mid].ct, dxl.ADDR_GOAL_POSITION, raw)
        self._sync_present()

    def _sync_present(self) -> None:
        for n in JOINTS:
            c = self.true_cal.joints[n]
            q = self.d.qpos[self.qadr[n]]
            dq = self.d.qvel[self.vadr[n]]
            raw = c.to_raw(q)
            raw_vel = int(round(c.sign * dq / (0.229 * 2 * np.pi / 60)))
            self.motors[MOTOR_ID[n]].set_present(raw, raw_vel)
            struct.pack_into("<H", self.motors[MOTOR_ID[n]].ct, dxl.ADDR_PRESENT_VOLTAGE, int(round(self.vin * 10)))

    def _on_write(self, addr: int, written: dict) -> None:
        if addr == dxl.ADDR_GOAL_POSITION:
            self.step(self.dt_ctrl)

    def _goal_sim(self, n: str) -> float:
        c = self.true_cal.joints[n]
        raw = self.motors[MOTOR_ID[n]].goal
        return c.to_sim(raw)

    def step(self, duration: float) -> None:
        n_sub = max(1, int(round(duration / self.m.opt.timestep)))
        new_goals = {n: self._goal_sim(n) for n in JOINTS}
        old_goals = self._prev_goals or new_goals
        self._prev_goals = new_goals
        n_delay = int(round(self.act_delay_s / self.m.opt.timestep))
        for k in range(n_sub):
            goals = old_goals if k < n_delay else new_goals
            for n in JOINTS:
                mot = self.motors[MOTOR_ID[n]]
                if not mot.torque:
                    self.d.ctrl[self.act[n]] = 0.0
                    continue
                q = self.d.qpos[self.qadr[n]]
                dq = self.d.qvel[self.vadr[n]]
                duty = np.clip(mot.kp * ERR_GAIN * (goals[n] - q), -1.0, 1.0)
                i = np.clip((self.vin * duty - KT * dq) / RES, -I_MAX, I_MAX)
                self.d.ctrl[self.act[n]] = KT * i
            mujoco.mj_step(self.m, self.d)
            if self.fixed_base:
                self.d.qpos[0:3] = [0.0, 0.0, 0.30]
                self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                self.d.qvel[0:6] = 0.0
        self.sim_time += n_sub * self.m.opt.timestep

    # -------------------------------------------------------------- views
    @property
    def trunk_pos(self) -> np.ndarray:
        return self.d.qpos[0:3].copy()

    @property
    def trunk_quat(self) -> np.ndarray:
        return self.d.qpos[3:7].copy()

    def upright(self) -> float:
        R = quat_to_mat(self.trunk_quat)
        return float(R[2, 2])


class FakeImuDevice:
    """Accelerometer/gyro at the MJCF 'imu' site, re-expressed in a (possibly wrong) chip frame."""

    def __init__(self, plant: Plant, true_mount=None, gyro_bias=(0.0, 0.0, 0.0), noise: float = 0.0, seed: int = 0,
                 static: bool = False):
        self.p = plant
        self.static = static   # True: accelerometer = gravity only (for posed, non-simulated tests)
        self.site_id = plant.m.site("imu").id
        site = plant.m.site("imu")
        site_q = site.quat.copy()
        # MJCF sensors report in the site frame; the chip frame is given by true_mount
        R_site = quat_to_mat(site_q)
        R_chip = quat_to_mat(true_mount) if true_mount is not None else R_site
        self.R_site_to_chip = R_chip.T @ R_site
        self.bias = np.array(gyro_bias, float)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def read(self):
        d = self.p.d
        if self.static:
            R_ws = d.site_xmat[self.site_id].reshape(3, 3)
            acc_site = R_ws.T @ np.array([0.0, 0.0, 1.0])
        else:
            acc_site = d.sensordata[self.p.s_acc:self.p.s_acc + 3] / 9.81
        gyr_site = d.sensordata[self.p.s_gyr:self.p.s_gyr + 3]
        acc = self.R_site_to_chip @ acc_site
        gyr = self.R_site_to_chip @ gyr_site + self.bias
        if self.noise:
            acc = acc + self.rng.normal(0, self.noise, 3)
            gyr = gyr + self.rng.normal(0, self.noise, 3)
        return acc, gyr
