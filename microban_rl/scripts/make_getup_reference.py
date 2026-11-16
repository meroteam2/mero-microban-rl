"""Build the Microban-GetUp reference motion from a keyframe-editor file.

1. Training copy of the keyframes: --src (getup_v6.lz4, not modified) with the
   pike_push -> fold interval stretched to --push-to-fold-s (later keyframes
   shifted by the same amount), saved as data/getup_v6_train.lz4.
2. data/getup_reference.npz, in MJLab joint order, 50 Hz:
   - t, q_ref [T, 19]: the editor's motor-target trajectory (linear in time
     between keyframes, as robot_keyframe_kit builds it);
   - tilt_ref, z_ref [T]: trunk tilt from vertical and trunk height. Up to the
     last keyframe the open-loop MJLab replay still follows (--physical-until),
     the replay's values; afterwards linear between the keyframes' root poses;
   - replay_*: the open-loop replay's full state at every step (root pose and
     velocity relative to the env origin, joint pos / vel) -- physically
     consistent start states for reference-state initialization up to
     --physical-until;
   - kf_*: the keyframes' kinematic poses (on the floor, at rest) and times,
     start states for the later part of the motion;
   - stage names / times for the progress report.

  uv run python scripts/make_getup_reference.py
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.getup_env_cfg import REFERENCE_FILE, TRAIN_KEYFRAMES
from mjlab.envs import ManagerBasedRlEnv

import replay_keyframes as R  # scripts/ is on sys.path when run as a script

MJCF_ORDER = ["head", "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "right_hip_yaw",
              "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
              "left_shoulder_pitch", "left_shoulder_roll", "left_elbow", "left_hip_yaw", "left_hip_roll",
              "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"]


@dataclass(frozen=True)
class Args:
  src: str = str(Path.home() / "mero/keyframe_editor/keyframes/getup_v6.lz4")
  push_to_fold_s: float = 0.8
  physical_until: str = "pike_push"
  """Last keyframe the open-loop replay reaches (its states / root are used up to here)."""
  hold_s: float = 3.0


def main(a: Args) -> None:
  data = joblib.load(a.src)
  train = copy.deepcopy(data)
  seq = list(train["timed_sequence"])
  names = [n for n, _ in seq]
  i = names.index("fold")
  assert names[i - 1] == "pike_push", seq
  shift = a.push_to_fold_s - (seq[i][1] - seq[i - 1][1])
  train["timed_sequence"] = seq[:i] + [(n, t + shift) for n, t in seq[i:]]
  train["motion_name"] = Path(TRAIN_KEYFRAMES).stem
  for k in ("time", "qpos", "motor_vel", "joint_vel", "body_pos", "body_quat", "body_lin_vel",
            "body_ang_vel", "site_pos", "site_quat", "action"):  # old playback, not this sequence
    if k in train:
      train[k] = np.zeros((0,) + np.asarray(train[k]).shape[1:], dtype=np.asarray(train[k]).dtype)
  Path(TRAIN_KEYFRAMES).parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(train, TRAIN_KEYFRAMES, compress="lz4")
  print(f"training keyframes: {TRAIN_KEYFRAMES}\n  sequence {train['timed_sequence']}")

  # Open-loop replay of the training copy (exact start, no noise), recording the state.
  ra = R.Args(path=TRAIN_KEYFRAMES, num_envs=1, noise=0.0, settle_s=1.0, hold_s=a.hold_s)
  env = ManagerBasedRlEnv(cfg=R.env_cfg(ra), device="cuda:0")
  robot = env.scene["robot"]
  jn = [n.split("/")[-1] for n in robot.joint_names]
  dt = env.step_dt
  targets, seq = R.editor_targets(train, list(robot.joint_names), dt)
  default = robot.data.default_joint_pos[0].cpu().numpy()
  settle = int(ra.settle_s / dt)
  t_end = seq[-1][1]
  sched = np.concatenate([np.repeat(targets[:1], settle, 0), targets])
  act = torch.tensor(sched - default, device=env.device, dtype=torch.float32)
  rec = {k: [] for k in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")}
  tilt, z = [], []
  with torch.inference_mode():
    env.reset()
    for s in range(len(sched)):
      env.step(act[s][None])
      if s + 1 < settle:
        continue
      d = robot.data
      rec["root_pos"].append((d.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
      rec["root_quat"].append(d.root_link_quat_w[0].cpu().numpy())
      rec["root_lin_vel"].append(d.root_link_lin_vel_w[0].cpu().numpy())
      rec["root_ang_vel"].append(d.root_link_ang_vel_w[0].cpu().numpy())
      rec["joint_pos"].append(d.joint_pos[0].cpu().numpy())
      rec["joint_vel"].append(d.joint_vel[0].cpu().numpy())
      tilt.append(float(R.trunk_angles(d.root_link_quat_w)["tilt"][0]))
      z.append(float(d.root_link_pos_w[0, 2]))
  rec = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
  t = np.arange(len(targets)) * dt  # replay state s corresponds to sequence time t (after settle)
  n = min(len(t), len(tilt))
  t, q_ref = t[:n], targets[:n].astype(np.float32)
  rec = {k: v[:n] for k, v in rec.items()}
  tilt, z = np.array(tilt[:n]), np.array(z[:n])

  kf = {k["name"]: k for k in train["keyframes"]}
  perm = [MJCF_ORDER.index(x) for x in jn]
  kf_names = [n_ for n_, _ in seq]
  kf_t = np.array([tt for _, tt in seq])

  def kf_tilt(name):
    q = torch.tensor(np.asarray(kf[name]["qpos"][3:7], dtype=np.float32))[None]
    return float(R.trunk_angles(q)["tilt"][0])

  phys_t = dict(seq)[a.physical_until]
  later = [(tt, kf_tilt(n_), float(kf[n_]["qpos"][2])) for n_, tt in seq if tt >= phys_t]
  lt, ltilt, lz = (np.array(x) for x in zip(*later))
  tilt_ref = np.where(t <= phys_t, tilt, np.interp(t, lt, ltilt)).astype(np.float32)
  z_ref = np.where(t <= phys_t, z, np.interp(t, lt, lz)).astype(np.float32)
  # Keyframe kinematic poses in MJLab joint order; root xy at the origin.
  kf_root = np.array([np.r_[0.0, 0.0, kf[n_]["qpos"][2], kf[n_]["qpos"][3:7]] for n_ in kf_names], dtype=np.float32)
  kf_joint = np.array([np.asarray(kf[n_]["qpos"][7:], dtype=np.float32)[perm] for n_ in kf_names])
  out = Path(REFERENCE_FILE)
  np.savez_compressed(
    out, t=t.astype(np.float32), q_ref=q_ref, tilt_ref=tilt_ref, z_ref=z_ref, joint_names=np.array(jn),
    t_end=np.float32(t_end), physical_until=np.float32(phys_t),
    kf_names=np.array(kf_names), kf_t=kf_t.astype(np.float32), kf_root=kf_root, kf_joint=kf_joint,
    **{f"replay_{k}": v for k, v in rec.items()},
  )
  print(f"reference: {out}  T={len(t)} steps ({t[-1]:.2f} s), physical states until {phys_t:.1f} s ({a.physical_until})")
  for n_, tt in seq:
    i_ = int(np.argmin(abs(t - tt)))
    print(f"  {n_:13s} t {tt:5.2f}  tilt_ref {math.degrees(tilt_ref[i_]):6.1f}  z_ref {z_ref[i_]:.3f}  "
          f"(replay tilt {math.degrees(tilt[i_]):6.1f} z {z[i_]:.3f})")
  # Symmetry of the reference (it should be mirror-symmetric).
  from microban_rl.symmetry import _MIRROR_JOINTS
  asym = max(abs(q_ref[:, jn.index(l)] - s * q_ref[:, jn.index(r)]).max()
             for l, (r, s) in _MIRROR_JOINTS.items() if l.startswith("left_"))
  print(f"  reference mirror asymmetry (max over time, joints): {asym:.4f} rad")


if __name__ == "__main__":
  main(tyro.cli(Args))
