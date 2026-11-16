"""Write the keyframe-editor start file for getting up from the back (supine).

Creates ~/mero/keyframe_editor/init/getup_back_init.lz4 (new file; nothing else is
touched) with the keyframes

  - supine: supine.py's pose (SUPINE_ROOT_QUAT, SUPINE_JOINT_POS) dropped and held
    with robot.py's PD (kp 0.88, kd 0.01, 0.64 Nm) for 2 s in the editor scene,
  - crouch, stand_target: copied unchanged from keyframes/getup_v6.lz4,

and the timed sequence supine (0 s) -> crouch (5 s) -> stand_target (6.5 s).

The supine root is moved to x = SUPINE_X (feet toward +x) so that, after sitting
up with the feet ~0.12-0.13 m in front of the hips (the closest the hip-pitch
limit of 90 deg allows), the feet are where they are in getup_v6's crouch
(x ~ 0.35); crouch / stand_target keep their x = 0.34 from getup_v6.

  cd ~/mero/microban_rl
  CUDA_VISIBLE_DEVICES= uv run python scripts/make_getup_back_init.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mujoco
import numpy as np

from microban_rl.robot import MICROBAN_ROOT, XL330_DAMPING, XL330_EFFORT_LIMIT, XL330_STIFFNESS
from microban_rl.supine import SUPINE_HEIGHT, SUPINE_JOINT_POS, SUPINE_ROOT_QUAT

EDITOR = MICROBAN_ROOT.parent / "keyframe_editor"
SCENE = EDITOR / "microban_scene.xml"
SOURCE = EDITOR / "keyframes" / "getup_v6.lz4"
OUT = EDITOR / "init" / "getup_back_init.lz4"
SETTLE_S = 2.0
SUPINE_X = 0.22
SEQUENCE = [("supine", 0.0), ("crouch", 5.0), ("stand_target", 6.5)]


def joint_qpos_addr(model: mujoco.MjModel) -> dict[str, int]:
  return {model.joint(j).name: model.jnt_qposadr[j] for j in range(model.njnt)
          if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}


def settle(model: mujoco.MjModel, pos, quat, joints: dict[str, float]) -> np.ndarray:
  """Drop the pose and hold it with robot.py's PD for SETTLE_S; returns qpos."""
  data = mujoco.MjData(model)
  data.qpos[:3], data.qpos[3:7] = pos, quat
  adr = joint_qpos_addr(model)
  for n, v in joints.items():
    data.qpos[adr[n]] = v
  target = np.array([joints[model.actuator(i).name] for i in range(model.nu)])
  qadr = np.array([adr[model.actuator(i).name] for i in range(model.nu)])
  vadr = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
  for _ in range(int(SETTLE_S / model.opt.timestep)):
    tau = XL330_STIFFNESS * (target - data.qpos[qadr]) - XL330_DAMPING * data.qvel[vadr]
    data.ctrl[:] = np.clip(tau, -XL330_EFFORT_LIMIT, XL330_EFFORT_LIMIT)
    mujoco.mj_step(model, data)
  print(f"supine settled: trunk z {data.qpos[2]:.4f} m, |qvel| max {np.abs(data.qvel).max():.4f}")
  return data.qpos.copy()


def main() -> None:
  if OUT.exists():
    raise SystemExit(f"{OUT} already exists; remove it first to regenerate.")
  model = mujoco.MjModel.from_xml_path(str(SCENE))
  qpos = settle(model, (SUPINE_X, 0.0, SUPINE_HEIGHT), SUPINE_ROOT_QUAT, SUPINE_JOINT_POS)
  qpos[0], qpos[1] = SUPINE_X, 0.0  # keep the start exactly at (SUPINE_X, 0)

  # Keyframe layout of make_keyframe_editor_files.py / the editor's saves.
  names = [model.actuator(i).name for i in range(model.nu)]
  adr = joint_qpos_addr(model)
  ui_joints = [model.joint(j).name for j in range(model.njnt)
               if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  supine = {
    "name": "supine",
    "motor_pos": np.array([qpos[adr[n]] for n in names], dtype=np.float32),
    "joint_pos": np.array([qpos[adr[n]] for n in ui_joints], dtype=np.float32),
    "qpos": qpos.astype(np.float32),
  }

  src = {k["name"]: k for k in joblib.load(SOURCE)["keyframes"]}
  keyframes = [supine] + [{k: (v.copy() if hasattr(v, "copy") else v) for k, v in src[n].items()}
                          for n in ("crouch", "stand_target")]
  for k in keyframes:
    assert k["qpos"].shape == (model.nq,) and k["motor_pos"].shape == (model.nu,), k["name"]

  init = {"keyframes": keyframes, "timed_sequence": list(SEQUENCE), "motion_name": "getup_back"}
  joblib.dump(init, OUT, compress="lz4")
  print(f"wrote {OUT}: {[k['name'] for k in keyframes]}, sequence {SEQUENCE}")


if __name__ == "__main__":
  main()
