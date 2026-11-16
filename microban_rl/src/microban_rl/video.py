"""Record a trained policy to an mp4 file.

Shared by all Microban tasks. Usable as a function (`record_policy_video`) or as
the `microban-record` CLI:

  uv run microban-record Microban-Crawl \
    --checkpoint-file logs/rsl_rl/microban_crawl/<run>/model_1500.pt \
    --output ~/mero/videos/crawl.mp4 --seconds 12

  # top view (camera straight down; the robot crawls left to right)
  uv run microban-record Microban-Crawl-Arms-2 --checkpoint-file <model.pt> \
    --output ~/mero/videos/crawl_top.mp4 --elevation -89.9 --azimuth 90

On WSL2, offscreen rendering goes through WSLg's GLFW (EGL is unavailable);
microban_rl/__init__.py selects that backend before mujoco is imported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mediapy as media
import torch
import tyro

import microban_rl  # noqa: F401  (registers the Microban tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def record_policy_video(
  task_id: str,
  checkpoint_file: str | Path,
  output: str | Path,
  seconds: float = 12.0,
  width: int = 1280,
  height: int = 720,
  device: str | None = None,
  elevation: float | None = None,
  azimuth: float | None = None,
  distance: float | None = None,
) -> Path:
  """Roll out one environment with a trained policy and save it as an mp4.

  Uses the task's play config (no observation noise, effectively endless
  episode) with a single environment and the task's tracking camera;
  elevation / azimuth (deg) and distance (m) override its angle.

  Returns:
    The path of the written video.
  """
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  output = Path(output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = width
  env_cfg.viewer.height = height
  env_cfg.viewer.max_extra_envs = 0
  for name, value in (("elevation", elevation), ("azimuth", azimuth), ("distance", distance)):
    if value is not None:
      setattr(env_cfg.viewer, name, value)
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  num_frames = int(round(seconds / env.step_dt))
  frames = []
  obs = vec_env.get_observations()
  with torch.inference_mode():
    for _ in range(num_frames):
      obs, _, _, _ = vec_env.step(policy(obs))
      frames.append(env.render())
  env.close()

  media.write_video(output, frames, fps=round(1.0 / env.step_dt))
  print(f"[INFO] Saved {len(frames)} frames ({seconds:.1f}s) to {output}")
  return output


@dataclass(frozen=True)
class RecordConfig:
  task_id: tyro.conf.Positional[str]
  """Registered task, e.g. Microban-Crawl."""
  checkpoint_file: str
  """Path to a model_*.pt checkpoint."""
  output: str = "videos/policy.mp4"
  seconds: float = 12.0
  width: int = 1280
  height: int = 720
  device: str | None = None
  elevation: float | None = None
  """Camera elevation in degrees (-89.9 = top view). Default: the task's camera."""
  azimuth: float | None = None
  """Camera azimuth in degrees (90 with a top view: +x points right)."""
  distance: float | None = None
  """Camera distance in meters."""


def main() -> None:
  cfg = tyro.cli(RecordConfig, description="Record a trained Microban policy to mp4.")
  record_policy_video(**asdict(cfg))


if __name__ == "__main__":
  main()
