"""Reward / contact-force / CPG diagnosis of a trained Microban crawl policy.

Rolls out the policy under the task's *training* reward terms (play config for
everything else) and prints:

- the mean contribution of every reward term (weight x value, per second),
  forward reward vs. the sum of penalties,
- how forward push relates to the contact-force threshold: share of the
  forward (+x) ground impulse delivered while a part's vertical force exceeds
  the threshold, and the horizontal force needed to drag the sliding parts,
- per-joint motion amplitude and mean offset (unused / raised limbs),
- for CPG tasks: learned frequency, L/R phase shift and per-joint CPG amplitude.

  uv run python scripts/diagnose_rewards.py Microban-Crawl-CPG \
      --checkpoint-file logs/rsl_rl/microban_crawl_cpg/<run>/model_700.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.crawl_env_cfg import (
  GROUND_CONTACT_SENSOR,
  GROUND_FORCE_SENSOR,
  contact_slip_speed,
  robot_weight,
)
from microban_rl.cpg import FREQ_RANGE
from microban_rl.curriculum import CurriculumState
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

SKIP_S = 2.0  # ignore the start-up transient (= CPG reward warm-up)


@dataclass(frozen=True)
class Args:
  task_id: tyro.conf.Positional[str]
  checkpoint_file: str
  num_envs: int = 256
  seconds: float = 12.0
  threshold: float = 1.5
  """Contact force threshold in body weights."""


def main(task_id: str, checkpoint_file: str, num_envs: int, seconds: float,
         threshold: float) -> None:
  device = "cuda:0"
  train_cfg = load_env_cfg(task_id, play=False)
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.rewards = train_cfg.rewards  # play cfg may drop terms; use training ones
  env_cfg.scene.num_envs = num_envs
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)
  env._penalty_curriculum = CurriculumState(scale=1.0)  # report full penalty weights

  robot = env.scene["robot"]
  force = env.scene[GROUND_FORCE_SENSOR.name]
  names = list(force.primary_names)
  weight = robot_weight(env)
  joint_names = list(robot.joint_names)
  rm = env.reward_manager
  term_names = list(rm.active_terms)
  cpg = (env.action_manager.get_term("cpg")
         if "cpg" in env.action_manager.active_terms else None)
  mj = env.sim.mj_model
  mu = float(np.max(mj.geom_friction[:, 0]))

  skip = int(SKIP_S / env.step_dt)
  rew, vx, q, fh, slip, contact, cpg_log, amp_log = [], [], [], [], [], [], [], []
  obs = vec_env.get_observations()
  with torch.inference_mode():
    for t in range(int(seconds / env.step_dt)):
      obs, _, _, _ = vec_env.step(policy(obs))
      if t < skip:
        continue
      rew.append(rm._step_reward.clone().cpu())  # weight x value (not x dt)
      vx.append(robot.data.root_link_lin_vel_w[:, 0].cpu())
      q.append((robot.data.joint_pos - robot.data.default_joint_pos).cpu())
      fh.append(force.data.force_history.clone().cpu())  # [B, P, H, 3] world
      s, c = contact_slip_speed(env)
      slip.append(s.cpu())
      contact.append(c.cpu())
      if cpg is not None:
        cpg_log.append(torch.stack([cpg.frequency, cpg.phase_shift], -1).cpu())
        amp_log.append(torch.stack([torch.hypot(cpg._a, cpg._b), cpg._o], -1).cpu())

  rew = torch.stack(rew).numpy()  # [T, N, K]
  vx = torch.stack(vx).numpy()
  print(f"\n=== {task_id}: {num_envs} envs, {seconds - SKIP_S:.0f} s after {SKIP_S:.0f} s ===")
  print(f"mean forward speed {vx.mean():.3f} m/s\n")
  print("reward contributions (weight x value, per second):")
  mean = rew.mean(axis=(0, 1))
  for k in np.argsort(-np.abs(mean)):
    print(f"  {term_names[k]:16s} {mean[k]:+8.3f}")
  fwd = mean[term_names.index("forward_vel")]
  low = mean[term_names.index("trunk_low")]
  pen = mean[[i for i, n in enumerate(term_names) if n not in ("forward_vel", "trunk_low")]]
  print(f"  forward_vel {fwd:+.3f} | trunk_low {low:+.3f} | penalties total {pen.sum():+.3f}"
        f" | net {mean.sum():+.3f}")

  # Contact forces. Orient so that +z is the floor pushing the robot up.
  f = torch.stack(fh).numpy()  # [T, N, P, H, 3]
  f = f * np.sign(np.nanmean(f[..., 2]) or 1.0)
  fz, fx = f[..., 2] / weight, f[..., 0] / weight
  push = np.clip(fx, 0, None)
  hi = fz > threshold
  print(f"\ncontact forces (W = {weight:.2f} N, friction mu = {mu:.2f}, threshold"
        f" {threshold} W):")
  print(f"  forward (+x) ground impulse delivered while Fz > {threshold} W: "
        f"{push[hi].sum() / max(push.sum(), 1e-9) * 100:.1f}%")
  print(f"  mean net +x ground force {fx.sum(axis=2).mean():+.3f} W, mean forward"
        f" push {push.sum(axis=2).mean():.3f} W, total Fz {fz.sum(axis=2).mean():.2f} W")
  print(f"  {'body':32s} {'Fz>thr %':>8s} {'Fz p99':>7s} {'mean +Fx':>8s}"
        f" {'+Fx share':>9s} {'Fx/Fz':>6s}")
  for i in np.argsort(-push.sum(axis=(0, 1, 3))[:]):
    c = fz[:, :, i] > 0.01
    if c.mean() < 0.005:
      continue
    share = push[:, :, i].sum() / max(push.sum(), 1e-9)
    ratio = np.abs(fx[:, :, i][c]).mean() / fz[:, :, i][c].mean()
    print(f"  {names[i]:32s} {hi[:, :, i][c].mean() * 100:7.1f}% "
          f"{np.percentile(fz[:, :, i][c], 99):6.2f}W {push[:, :, i].mean():7.3f}W "
          f"{share * 100:8.1f}% {ratio:6.2f}")

  # Joint motion: which limbs are used, which are held away from the pose.
  qa = torch.stack(q).numpy()
  amp, off = qa.std(axis=0).mean(axis=0), qa.mean(axis=(0, 1))
  print("\nper-joint motion (std rad | mean offset from default rad):")
  for j, n in enumerate(joint_names):
    print(f"  {n:22s} {amp[j]:.3f} | {off[j]:+.2f}")

  if cpg is not None:
    c = torch.stack(cpg_log).numpy()
    a = torch.stack(amp_log).numpy()  # [T, N, J, 2]
    print(f"\nCPG: frequency mean {c[..., 0].mean():.2f} Hz (p10 "
          f"{np.percentile(c[..., 0], 10):.2f}, p90 {np.percentile(c[..., 0], 90):.2f}),"
          f" at the {FREQ_RANGE[0]} / {FREQ_RANGE[1]} Hz limits"
          f" {np.mean(c[..., 0] < FREQ_RANGE[0] + 0.01) * 100:.0f}% /"
          f" {np.mean(c[..., 0] > FREQ_RANGE[1] - 0.01) * 100:.0f}%")
    print(f"     L/R phase shift mean {np.degrees(c[..., 1].mean()):.0f} deg (p10 "
          f"{np.degrees(np.percentile(c[..., 1], 10)):.0f}, p90 "
          f"{np.degrees(np.percentile(c[..., 1], 90)):.0f})")
    print("     commanded amplitude sqrt(a^2+b^2) | offset o (rad):")
    for j, n in enumerate(cpg.cpg_names):
      print(f"       {n:22s} {a[..., j, 0].mean():.3f} | {a[..., j, 1].mean():+.2f}")

  sl, ct = torch.stack(slip).numpy(), torch.stack(contact).numpy()
  print("\ncontact % | slip speed m/s:")
  for i in np.argsort(-ct.mean(axis=(0, 1))):
    if ct[..., i].mean() > 0.005:
      print(f"  {names[i]:32s} {ct[..., i].mean() * 100:5.1f}% | {sl[..., i][ct[..., i]].mean():.3f}")


if __name__ == "__main__":
  main(**asdict(tyro.cli(Args)))
