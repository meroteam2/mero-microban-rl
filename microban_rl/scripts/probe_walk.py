"""Print a few Microban-Walk envs over time under a fixed command (debugging).

uv run python scripts/probe_walk.py <model.pt> [--vx 0.2]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import tyro

import microban_rl  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import os

TASK = os.environ.get("TASK", "Microban-Walk")


def feet_contact(env):
  return (env.scene["feet_ground_contact"].data.found > 0).float()


@dataclass(frozen=True)
class Args:
  checkpoint: tyro.conf.Positional[str]
  vx: float = 0.2
  envs: int = 4
  seconds: float = 4.0


def main(a: Args) -> None:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = a.envs
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  ag = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  runner = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  env.reset()
  term.vel_command_b[:] = torch.tensor([a.vx, 0.0, 0.0], device=env.device)
  term.is_standing_env[:] = False
  obs = ve.get_observations()
  with torch.inference_mode():
    for s in range(int(a.seconds / env.step_dt)):
      act = policy(obs)
      obs, _, _, _ = ve.step(act)
      if s % 6 == 0:
        o = obs["actor"][0]
        print(f"t {s * env.step_dt:5.2f} cmd_obs {[round(v, 2) for v in o[60:63].tolist()]} "
              f"cmd_term {term.vel_command_b[0].tolist()} x {robot.data.root_link_pos_w[0, 0]:+.3f} "
              f"vx_b {robot.data.root_link_lin_vel_b[0, 0]:+.3f} contact {feet_contact(env)[0].tolist()} "
              f"|a| {act[0].abs().mean():.2f}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
