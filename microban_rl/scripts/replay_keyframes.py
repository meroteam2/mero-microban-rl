"""Stage 2-1: replay a keyframe-editor .lz4 open-loop in MJLab and report where it fails.

The motor targets are the editor's trajectory: linear interpolation of each
keyframe's motor_pos at the absolute times of timed_sequence, 50 Hz. They are
sent to robot.py's position actuators (kp 0.88, kd 0.01, 0.64 Nm torque limit)
through the 10.3 rad/s target speed limit (SpeedLimitedJointPositionAction, as
in Crawl-Arms / CPG 5). The robot starts prone (prone.py), holds the first
keyframe for --settle-s, plays the sequence, then holds the last keyframe for
--hold-s.

Everything is measured per environment (env 0 has no randomization; the others
use prone.py's reset randomization times --noise) and reported as
- a timeline: trunk tilt from vertical, trunk height, limbs on the floor,
  largest torque, how many motors are at the torque limit;
- per sequence segment: time at the torque limit per joint (where it is
  tightest), peak joint speed, target clipping by the speed limit, tracking error
  at the keyframe arrival;
- at the end: the stage-1 stand conditions (pitch 0..+0.2, |roll| < 0.1, both
  soles down, nearly still) and whether the robot flipped onto its back.

  uv run python scripts/replay_keyframes.py ~/mero/keyframe_editor/keyframes/getup_1111.lz4
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import joblib
import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.crawl_arms_env_cfg import MAX_TARGET_SPEED, SpeedLimitedJointPositionActionCfg
from microban_rl.prone import get_microban_prone_robot_cfg, reset_prone
from microban_rl.robot import MICROBAN_ARTICULATION, XL330_ACTUATOR_CFG, XL330_EFFORT_LIMIT, XL330_STIFFNESS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg

LIMBS = {  # body-name suffixes in the MJCF
  "R_foot": ("foot",), "L_foot": ("foot_2",),
  "R_hand": ("radius",), "L_hand": ("radius_2",),
  "knees": ("tibia__configuration_left", "tibia__configuration_right"),
  "trunk": ("trunk", "hip_block__configuration_left", "hip_block__configuration_right", "head",
            "shoulder", "shoulder_2"),
}
CONTACT_Z = 0.003  # lowest point within 3 mm of the floor = touching
SAT = 0.98  # |tau| >= SAT * limit = at the torque limit


@dataclass(frozen=True)
class Args:
  path: tyro.conf.Positional[str]
  num_envs: int = 256
  noise: float = 1.0
  """Scale of prone.py's reset randomization for envs 1.. (env 0 is exact)."""
  settle_s: float = 1.0
  hold_s: float = 2.0
  max_speed: float = MAX_TARGET_SPEED
  kp: float = XL330_STIFFNESS
  seed: int = 0
  timeline_dt: float = 0.5
  out: str = ""


def editor_targets(data: dict, joint_names: list[str], dt: float) -> tuple[np.ndarray, list]:
  """[T, J] motor targets (entity joint order) at dt, as robot_keyframe_kit builds them."""
  kf = {k["name"]: k for k in data["keyframes"]}
  seq = list(data["timed_sequence"])
  # motor_pos is in MJCF actuator order = MJCF hinge order (head, right_*, left_*).
  mjcf = ["head", "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "right_hip_yaw",
          "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "left_shoulder_pitch", "left_shoulder_roll", "left_elbow", "left_hip_yaw", "left_hip_roll",
          "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"]
  perm = [mjcf.index(n.split("/")[-1]) for n in joint_names]
  times = np.array([t for _, t in seq])
  assert np.all(np.diff(times) > 0), f"sequence times must increase: {seq}"
  pos = np.stack([np.asarray(kf[n]["motor_pos"], dtype=np.float64)[perm] for n, _ in seq])
  t = np.arange(0.0, times[-1] + 1e-9, dt)
  out = np.stack([np.interp(t, times, pos[:, j]) for j in range(pos.shape[1])], axis=1)
  return out, seq


def env_cfg(a: Args):
  from microban_rl.walk_onnx import walk_onnx_env_cfg

  cfg = walk_onnx_env_cfg(a.num_envs, kp=a.kp)
  robot = get_microban_prone_robot_cfg(
    articulation=replace(MICROBAN_ARTICULATION, actuators=(replace(XL330_ACTUATOR_CFG, stiffness=a.kp),))
  )
  cfg.scene.entities["robot"] = robot
  cfg.actions = {
    "joint_pos": SpeedLimitedJointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=True,
      max_speed=a.max_speed, ramp_steps=0,
    )
  }
  s = a.noise
  cfg.events = {
    "reset_prone": EventTermCfg(
      func=reset_prone, mode="reset",
      params={"yaw_range": (0.0, 0.0), "tilt_range": (-0.15 * s, 0.15 * s),
              "xy_range": (-0.05 * s, 0.05 * s), "joint_pos_range": (-0.15 * s, 0.15 * s)},
    )
  }
  return cfg


class Geometry:
  """Per-body lowest point above the floor (boxes, cylinders; as walk_onnx_check)."""

  def __init__(self, env):
    import mujoco

    mj = env.sim.mj_model
    box_t, cyl_t = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    g = [i for i in range(mj.ngeom) if (mj.geom_contype[i] or mj.geom_conaffinity[i])
         and int(mj.geom_type[i]) in (box_t, cyl_t) and mj.body(mj.geom_bodyid[i]).name != "world"]
    dev = env.device
    self.env, self.idx = env, torch.tensor(g, device=dev)
    self.size = torch.tensor(mj.geom_size[g], device=dev, dtype=torch.float32)
    self.is_box = torch.tensor([int(mj.geom_type[i]) == box_t for i in g], device=dev)
    body = [mj.body(mj.geom_bodyid[i]).name.split("/")[-1] for i in g]
    self.groups = {k: torch.tensor([i for i, b in enumerate(body) if b in v], device=dev)
                   for k, v in LIMBS.items()}
    for k, v in self.groups.items():
      assert len(v), k

  def lowest(self) -> dict[str, torch.Tensor]:
    d = self.env.sim.data
    pos = d.geom_xpos[:, self.idx]
    rz = d.geom_xmat[:, self.idx].reshape(pos.shape[0], len(self.idx), 3, 3)[:, :, 2, :]
    box = pos[..., 2] - (rz.abs() * self.size).sum(-1)
    cz = rz[..., 2].abs()
    cyl = pos[..., 2] - cz * self.size[:, 1] - self.size[:, 0] * (1 - cz**2).clamp(min=0).sqrt()
    z = torch.where(self.is_box, box, cyl)
    return {k: z[:, v].amin(1) for k, v in self.groups.items()}


def trunk_angles(quat: torch.Tensor) -> dict[str, torch.Tensor]:
  """Tilt of the trunk +z (pelvis->head) from vertical; pitch/roll as walk.onnx sees them;
  belly (trunk +x) world z: -1 prone, +1 on the back."""
  from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

  b = quat.shape[0]
  ex = torch.tensor([1.0, 0, 0], device=quat.device).expand(b, 3)
  ez = torch.tensor([0, 0, 1.0], device=quat.device).expand(b, 3)
  head = quat_apply(quat, ez)
  g = quat_apply_inverse(quat, -ez)  # projected gravity
  return {
    "tilt": torch.acos(head[:, 2].clamp(-1, 1)),
    "pitch": torch.atan2(-g[:, 0], -g[:, 2]).neg(),  # + = forward lean (head toward belly side)
    "roll": torch.atan2(g[:, 1], -g[:, 2]),
    "belly_z": quat_apply(quat, ex)[:, 2],
  }


def main(a: Args) -> None:
  torch.manual_seed(a.seed)
  data = joblib.load(a.path)
  env = ManagerBasedRlEnv(cfg=env_cfg(a), device="cuda:0")
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  dt = env.step_dt
  targets, seq = editor_targets(data, names, dt)
  default = robot.data.default_joint_pos[0].cpu().numpy()
  n, dev = a.num_envs, env.device

  with torch.inference_mode():
    env.reset()
    if a.noise:  # env 0 exact: re-run the reset event with zero ranges for it
      reset_prone(env, torch.tensor([0], device=dev), yaw_range=(0.0, 0.0), tilt_range=(0.0, 0.0),
                  xy_range=(0.0, 0.0), joint_pos_range=(0.0, 0.0))
      env.sim.forward()
  geo = Geometry(env)
  settle, hold = int(a.settle_s / dt), int(a.hold_s / dt)
  T = settle + len(targets) + hold
  sched = np.concatenate([np.repeat(targets[:1], settle, 0), targets, np.repeat(targets[-1:], hold, 0)])
  seq_t = np.arange(T) * dt - a.settle_s  # 0 = sequence start
  act = torch.tensor(sched - default, device=dev, dtype=torch.float32)
  sched_t = torch.tensor(sched, device=dev, dtype=torch.float32)

  log = {k: [] for k in ("tau", "qvel", "qpos", "clip", "tilt", "pitch", "roll", "belly_z", "z")}
  low = {k: [] for k in LIMBS}
  am = env.action_manager.get_term("joint_pos")
  with torch.inference_mode():
    for t in range(T):
      env.step(act[t].expand(n, -1))
      d = robot.data
      log["tau"].append(d.actuator_force.abs().cpu())
      log["qvel"].append(d.joint_vel.abs().cpu())
      log["qpos"].append(d.joint_pos.cpu())
      log["clip"].append((am._processed_actions - sched_t[t]).abs().cpu())
      for k, v in trunk_angles(d.root_link_quat_w).items():
        log[k].append(v.cpu())
      log["z"].append((d.root_link_pos_w[:, 2]).cpu())
      for k, v in geo.lowest().items():
        low[k].append(v.cpu())
  L = {k: torch.stack(v).numpy() for k, v in log.items()}  # [T, B, ...]
  Z = {k: torch.stack(v).numpy() for k, v in low.items()}
  C = {k: v < CONTACT_Z for k, v in Z.items()}
  lim = XL330_EFFORT_LIMIT
  short = [x.split("/")[-1].replace("right_", "R_").replace("left_", "L_").replace("shoulder", "sh")
           .replace("ankle", "ank") for x in names]

  # Per-env "event" checks.
  flipped = (L["belly_z"] > 0.0).any(0)  # rolled onto the back at any time
  def med(x):  # per-env values -> median and p10/p90
    return f"{np.median(x):+.3f} [{np.percentile(x, 10):+.3f},{np.percentile(x, 90):+.3f}]"

  print(f"\n== {a.path}\n   sequence {seq}\n   {n} envs (env 0 exact, others noise x{a.noise}), "
        f"kp {a.kp:.3f}, torque limit {lim:.3f} Nm, speed limit {a.max_speed:.1f} rad/s, "
        f"settle {a.settle_s}s, hold {a.hold_s}s")
  print("\n-- timeline (env 0 | median over envs). tilt = trunk from vertical (90 = lying), "
        "contacts of env 0, #sat = motors at the torque limit")
  print(f"{'t':>6} {'tilt0':>6} {'tiltM':>6} {'z0':>6} {'zM':>6} {'tau0':>5} {'#sat0':>5}  contacts(env0)")
  step = max(1, int(a.timeline_dt / dt))
  for t in range(0, T, step):
    c = " ".join(k for k in LIMBS if C[k][t, 0])
    print(f"{seq_t[t]:6.2f} {math.degrees(L['tilt'][t, 0]):6.1f} {math.degrees(np.median(L['tilt'][t])):6.1f} "
          f"{L['z'][t, 0]:6.3f} {np.median(L['z'][t]):6.3f} {L['tau'][t, 0].max():5.2f} "
          f"{(L['tau'][t, 0] >= SAT * lim).sum():5d}  {c}")

  print("\n-- per segment (all envs): share of time at the torque limit per joint (top 4), "
        "peak |qvel|, speed-limit clipping, tracking error at arrival (env 0 / median)")
  bounds = [(seq[i][0], seq[i + 1][0], seq[i][1], seq[i + 1][1]) for i in range(len(seq) - 1)]
  rows = []
  for k0, k1, t0, t1 in bounds:
    s = (seq_t >= t0) & (seq_t < t1)
    sat = (L["tau"][s] >= SAT * lim).mean((0, 1))
    top = np.argsort(-sat)[:4]
    arr = int(np.argmin(abs(seq_t - t1)))
    err = np.abs(L["qpos"][arr] - sched[arr]).max(1)  # per env, max over joints
    errj = int(np.argmax(np.abs(L["qpos"][arr, 0] - sched[arr])))
    row = {"segment": f"{k0}->{k1}", "t": [t0, t1],
           "sat_top": {short[j]: round(float(sat[j]), 3) for j in top},
           "qvel_peak": float(L["qvel"][s].max()), "clip_max": float(L["clip"][s].max()),
           "err_env0": float(err[0]), "err_env0_joint": short[errj], "err_median": float(np.median(err))}
    rows.append(row)
    print(f"{row['segment']:24s} {t0:4.1f}-{t1:4.1f}s  sat: " +
          ", ".join(f"{k} {v * 100:.0f}%" for k, v in row["sat_top"].items()) +
          f" | qvel {row['qvel_peak']:.1f} | clip {row['clip_max']:.2f} | err {row['err_env0']:.2f} "
          f"({row['err_env0_joint']}) / {row['err_median']:.2f}")

  # Keyframe arrival: achieved trunk tilt / height vs. the keyframe's own root pose.
  print("\n-- at each keyframe time: keyframe root (editor) vs. achieved (env 0 | median); contacts env 0")
  kf = {k["name"]: k for k in data["keyframes"]}
  for name, tk in seq:
    i = int(np.argmin(abs(seq_t - tk)))
    q = torch.tensor(np.asarray(kf[name]["qpos"][3:7], dtype=np.float32))[None]
    want = math.degrees(trunk_angles(q)["tilt"][0])
    c = " ".join(k for k in LIMBS if C[k][i, 0])
    print(f"  {name:13s} t {tk:4.1f}  tilt {want:5.1f} -> {math.degrees(L['tilt'][i, 0]):5.1f} | "
          f"{math.degrees(np.median(L['tilt'][i])):5.1f} deg   z {float(kf[name]['qpos'][2]):.3f} -> "
          f"{L['z'][i, 0]:.3f} | {np.median(L['z'][i]):.3f}   {c}")

  # Final stand check (stage-1 conditions), per env, at the end of the hold.
  e = -1
  feet = C["R_foot"][e] & C["L_foot"][e]
  still = L["qvel"][e].max(1) < 1.0
  pitch, roll = L["pitch"][e], L["roll"][e]
  stand = (pitch > -0.0) & (pitch < 0.2) & (np.abs(roll) < 0.1) & feet & ~C["trunk"][e] & ~C["knees"][e]
  arrive = int(np.argmin(abs(seq_t - seq[-1][1])))
  print(f"\n-- end of hold (t {seq_t[e]:.1f}s): stand conditions met {stand.mean() * 100:.1f}% "
        f"(+ still {(stand & still).mean() * 100:.1f}%), flipped onto back {flipped.mean() * 100:.1f}%")
  print(f"   pitch {med(pitch)}  roll {med(roll)}  tilt(deg) {med(np.degrees(L['tilt'][e]))}  "
        f"z {med(L['z'][e])}  both feet {feet.mean() * 100:.0f}%  trunk on floor {C['trunk'][e].mean() * 100:.0f}%")
  print(f"   at the last keyframe time (t {seq[-1][1]}): tilt(deg) {med(np.degrees(L['tilt'][arrive]))}  "
        f"z {med(L['z'][arrive])}")
  print(f"   peak torque {L['tau'].max():.3f} Nm, peak |qvel| {L['qvel'].max():.1f} rad/s")
  print("RESULT " + json.dumps({"path": a.path, "stand_rate": float(stand.mean()),
                                "flip_rate": float(flipped.mean()), "segments": rows}))
  if a.out:
    np.savez_compressed(a.out, seq_t=seq_t, sched=sched, names=np.array(names),
                        **{k: v for k, v in L.items()}, **{f"low_{k}": v for k, v in Z.items()})


if __name__ == "__main__":
  main(tyro.cli(Args))
