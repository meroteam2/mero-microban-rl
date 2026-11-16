"""Generate the robot_keyframe_kit (keyframe-editor) files for Microban get-up keyframes.

Writes to ~/mero/keyframe_editor/:

- microban_scene.xml: microban's scene.xml with
    * the MJCF torque motors limited to robot.py's torque limit (0.64 Nm),
    * MJLab's simulation options from the Stand / Crawl tasks (5 ms timestep,
      implicitfast, 10 solver / 20 line-search iterations),
    * "left_hand" / "right_hand" sites at the forearm tip cylinders (IK targets),
    * model keyframes "home" (= stand_target, the editor's startup pose) and
      "prone" (start pose, settled).
  Joint ranges, damping, friction and armature are the MJCF's (= robot.py).
- config.yaml: mirror pairs and signs from symmetry.py (the verified map), the
  editor's PD gains = robot.py's position actuator (kp 0.88, kd 0.01, torque
  limit 0.64), 20 ms control step = 4 x 5 ms substeps, end effectors.
- init/getup_init.lz4: keyframes "prone" and "stand_target" with the timed
  sequence prone (0 s) -> stand_target (3 s), to start editing from.

The prone pose is settled by simulating it under the same PD for 1.5 s.
stand_target = the walk.onnx start pose approved in stage 1 (default pose,
upright, both soles on the floor), placed kinematically at INIT_HEIGHT: held by
PD alone the default pose falls backward within ~1 s, so it cannot be settled.

  uv run python scripts/make_keyframe_editor_files.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mujoco
import numpy as np
import yaml

from microban_rl.prone import PRONE_HEIGHT, PRONE_JOINT_POS, PRONE_ROOT_QUAT
from microban_rl.robot import (
  DEFAULT_JOINT_POS,
  INIT_HEIGHT,
  MICROBAN_ROOT,
  XL330_DAMPING,
  XL330_EFFORT_LIMIT,
  XL330_STIFFNESS,
)
from microban_rl.symmetry import _MIRROR_JOINTS

OUT = MICROBAN_ROOT.parent / "keyframe_editor"
MJCF_DIR = MICROBAN_ROOT / "src" / "model" / "mjcf"
POLICY_DT = 0.02
TIMESTEP = 0.005
SETTLE_S = 1.5


def build_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MJCF_DIR / "scene.xml"))
  spec.meshdir = str(MJCF_DIR / "assets")
  spec.option.timestep = TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.iterations = 10
  spec.option.ls_iterations = 20
  for act in spec.actuators:
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-XL330_EFFORT_LIMIT, XL330_EFFORT_LIMIT]
  # Hand sites at the small (r 5.2 mm) forearm-tip collision cylinder.
  for body_name, side in (("radius_2", "left"), ("radius", "right")):
    body = spec.body(body_name)
    tip = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER and g.contype
           and g.size[0] < 0.007]
    assert len(tip) == 1, body_name
    body.add_site(name=f"{side}_hand", pos=tip[0].pos, group=3)
  return spec


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
  for _ in range(int(SETTLE_S / TIMESTEP)):
    tau = XL330_STIFFNESS * (target - data.qpos[qadr]) - XL330_DAMPING * data.qvel[vadr]
    data.ctrl[:] = np.clip(tau, -XL330_EFFORT_LIMIT, XL330_EFFORT_LIMIT)
    mujoco.mj_step(model, data)
  print(f"  settled: trunk z {data.qpos[2]:.4f} m, |qvel| max {np.abs(data.qvel).max():.4f}")
  return data.qpos.copy()


def place(model: mujoco.MjModel, pos, quat, joints: dict[str, float]) -> np.ndarray:
  """Kinematic pose; prints the floor contact distance of the lowest geom."""
  data = mujoco.MjData(model)
  data.qpos[:3], data.qpos[3:7] = pos, quat
  adr = joint_qpos_addr(model)
  for n, v in joints.items():
    data.qpos[adr[n]] = v
  mujoco.mj_forward(model, data)
  dist = [data.contact[i].dist for i in range(data.ncon)]
  print(f"stand_target: trunk z {data.qpos[2]:.4f} m, {data.ncon} floor contacts, "
        f"min distance {min(dist) * 1e3 if dist else float('nan'):+.2f} mm")
  return data.qpos.copy()


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  spec = build_spec()
  model = spec.compile()

  print("prone:")
  prone = settle(model, (0.0, 0.0, PRONE_HEIGHT), PRONE_ROOT_QUAT, PRONE_JOINT_POS)
  stand = place(model, (0.0, 0.0, INIT_HEIGHT), (1.0, 0.0, 0.0, 0.0), DEFAULT_JOINT_POS)

  # "home" is the editor's startup pose. It must not be the prone pose: the
  # editor builds its root roll/pitch/yaw sliders (range +-3.14) from it, and
  # lying face-down is pitch ~90 deg, where the Euler roll comes out at ~pi.
  spec.add_key(name="home", qpos=stand)
  spec.add_key(name="prone", qpos=prone)
  scene = OUT / "microban_scene.xml"
  scene.write_text(spec.to_xml())
  model = mujoco.MjModel.from_xml_path(str(scene))  # check it loads back

  # Mirror map of symmetry.py: left joint -> (right joint, sign); the editor sets
  # partner = sign * value, for both directions.
  pairs = {n: p for n, (p, s) in _MIRROR_JOINTS.items() if n.startswith("left_")}
  signs = {n: int(s) for n, (p, s) in _MIRROR_JOINTS.items() if n != "head"}
  config = {
    "name": "microban",
    "root_body": "trunk",
    "end_effectors": ["left_foot", "right_foot", "left_hand", "right_hand"],
    "mirror_pairs": pairs,
    "mirror_signs": signs,
    "dt": POLICY_DT,
    "save_dir": str(OUT / "keyframes"),
    "n_frames": round(POLICY_DT / TIMESTEP),
    "physics": {
      "kp": XL330_STIFFNESS,
      "kd": XL330_DAMPING,
      "motor_tau_limit": XL330_EFFORT_LIMIT,
      "dt": POLICY_DT,
    },
    "scene": {"auto_inject_floor": False, "show_floor": True},
  }
  (OUT / "config.yaml").write_text(
    "# Generated by microban_rl/scripts/make_keyframe_editor_files.py\n"
    + yaml.safe_dump(config, sort_keys=False)
  )

  names = [model.actuator(i).name for i in range(model.nu)]
  adr = joint_qpos_addr(model)
  ui_joints = [model.joint(j).name for j in range(model.njnt)
               if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

  def keyframe(name: str, qpos: np.ndarray) -> dict:
    return {
      "name": name,
      "motor_pos": np.array([qpos[adr[n]] for n in names], dtype=np.float32),
      "joint_pos": np.array([qpos[adr[n]] for n in ui_joints], dtype=np.float32),
      "qpos": qpos.astype(np.float32),
    }

  init = {
    "keyframes": [keyframe("prone", prone), keyframe("stand_target", stand)],
    "timed_sequence": [("prone", 0.0), ("stand_target", 3.0)],
    "motion_name": "getup",
  }
  (OUT / "init").mkdir(exist_ok=True)
  joblib.dump(init, OUT / "init" / "getup_init.lz4", compress="lz4")
  print(f"wrote {scene}, {OUT / 'config.yaml'}, {OUT / 'init' / 'getup_init.lz4'}")


if __name__ == "__main__":
  main()
