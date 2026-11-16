"""walk.onnx in microban's own simulator setup, headless (reference for walk_onnx_check.py).

Reproduces microban/src/sim/mujoco_controller.py + moves/walk.py without the
viewer: CPU MuJoCo on microban's scene.xml, BAM XL330 m6 motor model (kp
KP_DEFAULT while standing, KP_RL once walking, vin 7.5 V, 1.75 A limit), sensor
delays from sim_main.py, 50 Hz, spawn in NEUTRAL_POSE at z = 0.165, then
--settle-s holding NEUTRAL_POSE before walk.onnx takes over.

  uv run python scripts/walk_microban_sim.py --vx 0.3

Requires the BAM version microban's simulator is written against: MujocoController
with vin_drop_gain / max_current, i.e. the Rhoban/bam branch mjlab_qfrc_constraint
(commit 3dff2f27). Neither the PyPI release (1.0.3) nor bam main has that API, and
the branch is not installed in this project yet, so this script does not run as is.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
import onnxruntime as ort
import tyro

from microban_rl.robot import MICROBAN_ROOT

sys.path.insert(0, str(MICROBAN_ROOT / "src"))
from bam.model import load_model  # noqa: E402
from bam.mujoco import MujocoController  # noqa: E402
from constants import (  # noqa: E402
  BAM_MAX_CURRENT, BAM_VIN, BAM_VIN_MIN, BAM_VOLTAGE_DROP_GAIN, KP_DEFAULT, KP_RL,
  MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER,
)
from constants import IMU_MOUNT_QUAT  # noqa: E402


# microban/src/imu_reader.py (copied: importing it needs the 'ahrs' IMU package).
def imu_quat_to_body(q):
  w1, x1, y1, z1 = q
  w2, x2, y2, z2 = IMU_MOUNT_QUAT[0], -IMU_MOUNT_QUAT[1], -IMU_MOUNT_QUAT[2], -IMU_MOUNT_QUAT[3]
  return (
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  )


def quat_apply_inverse(quat, vec):
  xyz, w = np.array(quat[1:]), quat[0]
  t = 2 * np.cross(xyz, vec)
  return vec - w * t + np.cross(xyz, t)


@dataclass(frozen=True)
class Args:
  vx: float = 0.3
  settle_s: float = 1.0
  seconds: float = 10.0
  delays: bool = True


def main(a: Args) -> None:
  model = mujoco.MjModel.from_xml_path(str(MICROBAN_ROOT / "src/model/mjcf/scene.xml"))
  data = mujoco.MjData(model)
  names = list(MOTOR_TO_ID)
  qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names}
  vadr = {n: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names}
  data.qpos[2] = 0.165
  for n, v in NEUTRAL_POSE.items():
    data.qpos[qadr[n]] = v
  mujoco.mj_forward(model, data)
  bam_model = load_model(motor_name="xl330", model="m6")
  bam_model.actuator.kp = KP_DEFAULT
  bam_model.actuator.vin = BAM_VIN
  bam = MujocoController(bam_model, names, model, data, vin_drop_gain=BAM_VOLTAGE_DROP_GAIN,
                         vin_min=BAM_VIN_MIN, max_current=BAM_MAX_CURRENT)
  bam.reset(data.qpos)
  s_quat = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")]
  s_gyro = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")]
  d = {"vel": 1, "gyro": 3, "quat": 4, "act": 2} if a.delays else {"vel": 0, "gyro": 0, "quat": 0, "act": 0}
  buf = {k: deque(maxlen=v + 1) for k, v in d.items()}

  def delayed(k, x):
    if not buf[k]:
      buf[k].extend([x] * buf[k].maxlen)
    buf[k].append(x)
    return buf[k][0]

  sess = ort.InferenceSession(str(MICROBAN_ROOT / "src/agents/walk.onnx"))
  meta = sess.get_modelmeta().custom_metadata_map
  default = dict(zip(meta["joint_names"].split(","), map(float, meta["default_joint_pos"].split(","))))
  last = np.zeros(18, dtype=np.float32)
  target = dict(NEUTRAL_POSE)
  settle = int(a.settle_s / 0.02)
  fallen, fall_t, start = False, None, None
  steps_per_tick = round(0.02 / model.opt.timestep)
  for t in range(settle + int(a.seconds / 0.02)):
    q = {n: data.qpos[qadr[n]] for n in names}
    dq = delayed("vel", {n: data.qvel[vadr[n]] for n in names})
    gyro = delayed("gyro", np.array(data.sensordata[s_gyro:s_gyro + 3]))
    quat = delayed("quat", np.array(data.sensordata[s_quat:s_quat + 4]))
    grav = quat_apply_inverse(list(imu_quat_to_body(list(quat))), [0.0, 0.0, -1.0])
    if t == settle:
      bam.model.actuator.kp = KP_RL  # WalkMove.on_start
      start = (data.qpos[:2].copy(), data.qpos[3:7].copy())
    if t >= settle and grav[2] <= -0.5:
      cmd = [a.vx, 0.0, 0.0]
      obs = [*gyro, *grav, *(q[n] - default[n] for n in OBSERVATION_DOF_ORDER),
             *(dq[n] for n in OBSERVATION_DOF_ORDER), *last, *cmd]
      last = sess.run(None, {"obs": np.array([obs], dtype=np.float32)})[0][0]
      target = dict(NEUTRAL_POSE)
      for i, n in enumerate(OBSERVATION_DOF_ORDER):
        target[n] = default[n] + last[i]
    if grav[2] > -0.5 or data.qpos[2] < 0.10:
      if not fallen:
        fall_t = t * 0.02
      fallen = True
    for _ in range(steps_per_tick):
      tgt = delayed("act", dict(target))
      for n in names:
        bam.set_q_target(n, tgt[n])
      bam.update()
      mujoco.mj_step(model, data)

  p0, q0 = start
  disp = data.qpos[:2] - p0
  yaw = lambda qq: np.arctan2(2 * (qq[0] * qq[3] + qq[1] * qq[2]), 1 - 2 * (qq[2] ** 2 + qq[3] ** 2))  # noqa: E731
  h0 = yaw(q0)
  fwd = disp[0] * np.cos(h0) + disp[1] * np.sin(h0)
  lat = -disp[0] * np.sin(h0) + disp[1] * np.cos(h0)
  dyaw = np.degrees(np.angle(np.exp(1j * (yaw(data.qpos[3:7]) - h0))))
  print("RESULT " + json.dumps({"sim": "microban(BAM)", "vx": a.vx, "delays": a.delays,
                                "fallen": fallen, "fall_t": fall_t,
                                "fwd_speed": fwd / a.seconds, "lateral_m": lat,
                                "yaw_change_deg": dyaw}), flush=True)


if __name__ == "__main__":
  main(tyro.cli(Args))
