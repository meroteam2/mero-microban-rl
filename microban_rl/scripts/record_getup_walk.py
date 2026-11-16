"""Record the stage-4 sequence (get up from prone -> 1 s hold -> walk.onnx) to an mp4.

Same controller as scripts/getup_walk.py (GetUpThenWalk: get-up policy, hold
with the policy balancing around stand_target, gain 0.88 -> 0.275, walk.onnx
with heading hold), one env, prone start facing world +x (no heading
randomization) so that a fixed side camera sees the robot from its side; the
camera follows the trunk.

  uv run python scripts/record_getup_walk.py \
    --checkpoint logs/rsl_rl/microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt \
    --output ~/mero/videos/getup_walk.mp4
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mediapy as media
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.robot import XL330_DAMPING, XL330_STIFFNESS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr import pd_gains
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import getup_walk as G  # scripts/ is on sys.path when run as a script


@dataclass(frozen=True)
class Args:
  checkpoint: str
  output: str = "~/mero/videos/getup_walk.mp4"
  task: str = "Microban-GetUp-5"
  seconds: float = 20.0
  vx: float = 0.15
  heading_gain: float = 1.0
  hold_s: float = 1.0
  width: int = 1280
  height: int = 720
  elevation: float = -10.0
  azimuth: float = 90.0
  distance: float = 1.0
  seed: int = 3


def main(a: Args) -> None:
  torch.manual_seed(a.seed)
  out = Path(a.output).expanduser().resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}
  cfg.episode_length_s = 1e9
  pp = cfg.events["reset_getup"].params["prone_params"]
  pp["yaw_range"] = (0.0, 0.0)
  cfg.events["gains"] = EventTermCfg(
    func=pd_gains, mode="startup",
    params={"kp_range": (XL330_STIFFNESS, XL330_STIFFNESS), "kd_range": (XL330_DAMPING, XL330_DAMPING),
            "operation": "abs"})
  cfg.viewer.width, cfg.viewer.height, cfg.viewer.max_extra_envs = a.width, a.height, 0
  cfg.viewer.elevation, cfg.viewer.azimuth, cfg.viewer.distance = a.elevation, a.azimuth, a.distance
  ag = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  runner = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=env.device)
  ga = G.Args(checkpoint=a.checkpoint, task=a.task, num_envs=1, vx=a.vx, heading_gain=a.heading_gain,
              hold_s=a.hold_s, walk_s=1e9, hold_mode="policy")
  ctl = G.GetUpThenWalk(env, runner.get_inference_policy(device=env.device), ga)
  frames, phases = [], []
  with torch.inference_mode():
    env.reset()
    obs = ve.get_observations()
    for _ in range(int(round(a.seconds / env.step_dt))):
      phases.append(int(ctl.phase()[0]))
      obs, _, _, _ = ve.step(ctl(obs))
      frames.append(env.render())
    fell = bool(env.scene["robot"].data.projected_gravity_b[0, 2] > G.FALL_GRAVITY_Z)
  env.close()
  media.write_video(out, frames, fps=round(1.0 / env.step_dt))
  dt = 1.0 / round(1.0 / env.step_dt)
  print(f"[INFO] saved {len(frames)} frames ({len(frames) * dt:.1f} s) to {out}; get-up {phases.count(0) * dt:.1f} s, "
        f"hold {phases.count(1) * dt:.1f} s, walk {phases.count(2) * dt:.1f} s; upright at the end: {not fell}")


if __name__ == "__main__":
  main(tyro.cli(Args))
