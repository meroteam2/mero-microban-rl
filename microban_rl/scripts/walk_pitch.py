"""Trunk pitch, motor current and joint tracking while walking, per command (sim reference for real walk logs).

Reports, per forward command: mean trunk pitch (deg, + = forward) from the
projected gravity, and mean / p90 |q - target| of the leg pitch joints, to
compare with mero_walk/logs/walk_*.csv.

  TASK=Microban-Walk uv run python scripts/walk_pitch.py <model.pt>
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = os.environ.get("TASK", "Microban-Walk")
JOINTS = ("right_hip_pitch", "left_hip_pitch", "right_knee", "left_knee", "right_ankle_pitch", "left_ankle_pitch")


@dataclass(frozen=True)
class Args:
  checkpoint: tyro.conf.Positional[str]
  vx: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3)
  envs_per_command: int = 32
  seconds: float = 8.0


def main(a: Args) -> None:
  cfg = load_env_cfg(TASK, play=True)
  n = a.envs_per_command * len(a.vx)
  cfg.scene.num_envs = n
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  ag = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  runner = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  cmd = torch.zeros(n, 3, device=env.device)
  cmd[:, 0] = torch.tensor(a.vx, device=env.device).repeat_interleave(a.envs_per_command)
  jid = [list(robot.joint_names).index(j) for j in JOINTS]
  env.reset()
  term.vel_command_b[:] = cmd
  term.is_standing_env[:] = False
  obs = ve.get_observations()
  pitch, err, vel, amps, amps_peak = [], [], [], [], []
  KT = 0.366  # Nm/A, XL330 (microban BAM constants)
  with torch.inference_mode():
    for s in range(int(a.seconds / env.step_dt)):
      term.vel_command_b[:] = cmd
      obs, _, _, _ = ve.step(policy(obs))
      if s * env.step_dt > 2.0:
        g = robot.data.projected_gravity_b
        pitch.append(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])).cpu().numpy())
        e = robot.data.joint_pos[:, jid] - robot.data.joint_pos_target[:, jid]
        err.append(torch.rad2deg(e).cpu().numpy())
        vel.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
        cur = robot.data.actuator_force.abs().sum(1) / KT  # A, all 19 motors (copper current ~ |torque| / kt)
        amps.append(cur.cpu().numpy())
  pitch, err, vel, amps = np.array(pitch), np.array(err), np.array(vel), np.array(amps)
  print(f"{TASK} {a.checkpoint}")
  print(f"{'cmd vx':>7} | {'vx':>6} | pitch mean (p10, p90) deg | current A mean / p99 (sum of 19 motors, at control steps) | " + " | ".join(f"{j[:14]} err mean/p90" for j in JOINTS))
  for i, v in enumerate(a.vx):
    sl = slice(i * a.envs_per_command, (i + 1) * a.envs_per_command)
    p = pitch[:, sl]
    e = err[:, sl]
    cols = " | ".join(f"{e[..., k].mean():+6.1f}/{np.percentile(np.abs(e[..., k]), 90):5.1f}" for k in range(len(JOINTS)))
    cu = amps[:, sl]
    print(f"{v:7.2f} | {vel[:, sl].mean():6.3f} | {p.mean():5.1f} ({np.percentile(p, 10):5.1f}, {np.percentile(p, 90):5.1f}) | "
          f"{cu.mean():5.2f} / {np.percentile(cu, 99):5.2f} | {cols}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
