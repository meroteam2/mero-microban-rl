"""Summarize how a trained Microban policy moves, without rendering video.

Rolls out many environments (play config: no observation noise) and prints
speed, trunk height/attitude, which joints and limbs move most, left/right
phase coupling, gait frequency and which body parts touch the ground.

  uv run python scripts/analyze_policy.py Microban-Crawl \
      --checkpoint-file logs/rsl_rl/microban_crawl/<run>/model_1499.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.crawl_env_cfg import GROUND_CONTACT_SENSOR, contact_slip_speed
from microban_rl.prone import belly_direction_w, head_direction_w
from microban_rl.symmetry import _MIRROR_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

LIMBS = {
  "left arm": ("left_shoulder_pitch", "left_shoulder_roll", "left_elbow"),
  "right arm": ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow"),
  "left leg": ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
               "left_ankle_pitch", "left_ankle_roll"),
  "right leg": ("right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                "right_knee", "right_ankle_pitch", "right_ankle_roll"),
  "head": ("head",),
}


@dataclass(frozen=True)
class Args:
  task_id: tyro.conf.Positional[str]
  checkpoint_file: str
  num_envs: int = 256
  seconds: float = 10.0


def main(task_id: str, checkpoint_file: str, num_envs: int, seconds: float) -> None:
  device = "cuda:0"
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs
  sensors = tuple(env_cfg.scene.sensors or ())
  if not any(c.name == GROUND_CONTACT_SENSOR.name for c in sensors):
    env_cfg.scene.sensors = sensors + (GROUND_CONTACT_SENSOR,)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  sensor = env.scene["ground_contact"]
  joint_names = list(robot.joint_names)
  steps = int(seconds / env.step_dt)

  start_xy = robot.data.root_link_pos_w[:, :2].clone()
  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  q_log, z_log, vx_log, pitch_log, contact_log, act_log = [], [], [], [], [], []
  slip_log, cpg_log = [], []
  cpg = (env.action_manager.get_term("cpg")
         if "cpg" in env.action_manager.active_terms else None)
  obs = vec_env.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, dones, _ = vec_env.step(policy(obs))
      alive &= ~dones.bool()
      d = robot.data
      q_log.append((d.joint_pos - d.default_joint_pos).cpu())
      act_log.append(env.action_manager.action.cpu())
      z_log.append(d.root_link_pos_w[:, 2].cpu())
      vx_log.append(d.root_link_lin_vel_w[:, 0].cpu())
      head = head_direction_w(robot)
      pitch_log.append(torch.rad2deg(torch.asin(head[:, 2].clamp(-1, 1))).cpu())
      speed, in_contact = contact_slip_speed(env)
      slip_log.append(speed.cpu())
      contact_log.append(in_contact.float().cpu())
      if cpg is not None:
        cpg_log.append(torch.stack([cpg.frequency, cpg.phase_shift], -1).cpu())
    end_xy = robot.data.root_link_pos_w[:, :2]
    belly_z = belly_direction_w(robot)[:, 2]
    head_xy = head_direction_w(robot)[:, :2]

  a = alive.cpu().numpy()
  disp = (end_xy - start_xy).cpu().numpy()
  q = torch.stack(q_log).numpy()  # [T, N, J]
  z = torch.stack(z_log).numpy()
  vx = torch.stack(vx_log).numpy()
  pitch = torch.stack(pitch_log).numpy()
  contact = torch.stack(contact_log).numpy()  # [T, N, P]
  acts = torch.stack(act_log).numpy()
  slip = torch.stack(slip_log).numpy()  # [T, N, P] contact-point speed

  print(f"\n=== {task_id}: {num_envs} envs x {seconds:.0f}s ===")
  print(f"survived (no flip): {a.mean() * 100:.1f}%")
  print(f"forward distance  : mean {disp[a, 0].mean():.3f} m  "
        f"(speed {disp[a, 0].mean() / seconds:.3f} m/s), "
        f"p10/p90 {np.percentile(disp[a, 0], 10):.3f}/{np.percentile(disp[a, 0], 90):.3f}")
  print(f"lateral drift     : mean |y| {np.abs(disp[a, 1]).mean():.3f} m")
  yaw = np.degrees(np.arctan2(head_xy[:, 1].cpu().numpy(), head_xy[:, 0].cpu().numpy()))
  print(f"final heading err : mean |yaw| {np.abs(yaw[a]).mean():.1f} deg")
  print(f"trunk height      : mean {z.mean():.3f} m, max {z.max():.3f} m, "
        f"below 0.085 m {np.mean(z < 0.085) * 100:.1f}% of time")
  print(f"head pitch        : mean {pitch.mean():+.1f} deg (+ = head raised)")
  print(f"belly down (end)  : belly_z mean {belly_z.mean().item():+.2f} (-1 = flat prone)")
  print(f"inst. vx std      : {vx.std():.3f} m/s (lunging if >> mean)")

  amp = q.std(axis=0).mean(axis=0)  # per-joint motion amplitude
  print("\nper-limb motion (mean joint std, rad) and mean |action|:")
  for limb, names in LIMBS.items():
    idx = [joint_names.index(n) for n in names]
    a_txt = (f"   |a|={np.abs(acts[..., idx]).mean():.2f}"
             if acts.shape[-1] == len(joint_names) else "")
    print(f"  {limb:10s} {amp[idx].mean():.3f}{a_txt}")
  print("most active joints:")
  for j in np.argsort(-amp)[:8]:
    print(f"  {joint_names[j]:22s} std {amp[j]:.3f} rad, mean offset "
          f"{q[..., j].mean():+.2f} rad")

  # Dominant gait frequency and left/right coupling of the most active joints.
  dt = env.step_dt
  freqs = np.fft.rfftfreq(q.shape[0], dt)
  spec = np.abs(np.fft.rfft(q - q.mean(0), axis=0)).mean(axis=1)  # [F, J]
  top = np.argsort(-amp)[:4]
  f_dom = freqs[1:][np.argmax(spec[1:, top].sum(1))]
  print(f"\ndominant motion frequency: {f_dom:.2f} Hz")
  for base in ("shoulder_pitch", "elbow", "hip_pitch", "knee", "hip_roll"):
    li, ri = joint_names.index(f"left_{base}"), joint_names.index(f"right_{base}")
    c = np.mean([np.corrcoef(q[:, n, li], q[:, n, ri])[0, 1] for n in range(min(64, num_envs))])
    print(f"  L/R corr {base:15s} {c:+.2f}  (+1 in phase, -1 alternating)")
  ai, li = joint_names.index("left_shoulder_pitch"), joint_names.index("left_hip_pitch")
  c = np.mean([np.corrcoef(q[:, n, ai], q[:, n, li])[0, 1] for n in range(min(64, num_envs))])
  print(f"  arm/leg corr (L shoulder vs L hip pitch) {c:+.2f}")

  # Left/right symmetry: compare each left joint with its mirrored right partner
  # (right joint x mirror sign), so a perfectly mirror-symmetric gait gives 0.
  half = q[q.shape[0] // 2 :]  # skip the start-up transient
  amp_asym, pose_asym, weights = [], [], []
  for name, (partner, sign) in _MIRROR_JOINTS.items():
    if not name.startswith("left_"):
      continue
    li, ri = joint_names.index(name), joint_names.index(partner)
    al, ar = half[..., li].std(0), half[..., ri].std(0)  # [N]
    amp_asym.append(np.mean(np.abs(al - ar) / np.maximum((al + ar) / 2, 1e-3)))
    pose_asym.append(np.mean(np.abs(half[..., li].mean(0) - sign * half[..., ri].mean(0))))
    weights.append(np.mean(al + ar))
  w = np.array(weights) / np.sum(weights)
  print(f"\nL/R symmetry: amplitude mismatch {np.dot(w, amp_asym) * 100:.1f}% "
        f"(amplitude-weighted), mean posture mismatch {np.mean(pose_asym):.3f} rad")

  # Left/right phase difference of the leg pitch joints at the gait frequency.
  hf = np.fft.rfftfreq(half.shape[0], dt)
  k = 1 + np.argmin(np.abs(hf[1:] - f_dom))
  diffs, wts = [], []
  for base in ("hip_pitch", "knee", "ankle_pitch"):
    li, ri = joint_names.index(f"left_{base}"), joint_names.index(f"right_{base}")
    sign = _MIRROR_JOINTS[f"left_{base}"][1]
    zl = np.fft.rfft(half[..., li] - half[..., li].mean(0), axis=0)[k]
    zr = np.fft.rfft(sign * (half[..., ri] - half[..., ri].mean(0)), axis=0)[k]
    diffs.append(np.angle(zr * np.conj(zl)))
    wts.append(np.abs(zl) * np.abs(zr))
  d_all, w_all = np.concatenate(diffs), np.concatenate(wts)
  mean_phase = np.angle(np.sum(w_all * np.exp(1j * d_all)))
  spread = 1 - np.abs(np.sum(w_all * np.exp(1j * d_all))) / np.sum(w_all)
  print(f"L/R leg phase difference (measured at {hf[k]:.2f} Hz): "
        f"{np.degrees(np.abs(mean_phase)):.0f} deg (180 = alternating), "
        f"circular spread {spread:.2f}")

  if cpg_log:
    c = torch.stack(cpg_log).numpy()[q.shape[0] // 2 :]  # [T, N, 2]
    print(f"CPG (2nd half): frequency mean {c[..., 0].mean():.2f} Hz "
          f"(p10 {np.percentile(c[..., 0], 10):.2f}, p90 {np.percentile(c[..., 0], 90):.2f}), "
          f"L/R phase shift mean {np.degrees(c[..., 1].mean()):.0f} deg "
          f"(p10 {np.degrees(np.percentile(c[..., 1], 10)):.0f}, "
          f"p90 {np.degrees(np.percentile(c[..., 1], 90)):.0f})")

  # Horizontal speed of the contact point itself: ~0 means a planted contact,
  # close to the trunk speed means the part is dragged / sliding.
  print("\nground contact (% of time) | contact-point slip speed (m/s):")
  names = sensor.primary_names
  for i in np.argsort(-contact.mean(axis=(0, 1))):
    c = contact[..., i] > 0
    v = f"{slip[..., i][c].mean():.3f}" if c.any() else "  -  "
    print(f"  {names[i]:34s} {contact[..., i].mean() * 100:5.1f}%   {v}")


if __name__ == "__main__":
  main(**asdict(tyro.cli(Args)))
