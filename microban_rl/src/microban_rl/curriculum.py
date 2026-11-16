"""Speed-gated penalty curriculum for the crawl tasks.

A single scale s in [0, 1] multiplies the curriculum reward terms (and, via
CurriculumPpo, the symmetry mirror loss). It starts at `initial_scale` (0 for
training) and only grows while the batch-mean forward speed is above
`speed_threshold`, by `ramp_rate` per second of simulated time; below the
threshold it is held. So the policy first learns to move forward with only the
forward reward and the basic heading terms, and the shaping penalties are faded
in once it does.

The speed is an exponential moving average of the mean trunk +x velocity over
all environments that are past the reward warm-up.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import torch
from rsl_rl.algorithms import PPO

from mjlab.envs import ManagerBasedRlEnv


@dataclass
class CurriculumState:
  scale: float
  speed: float = 0.0


@dataclass(frozen=True)
class PenaltyCurriculumCfg:
  initial_scale: float = 0.0
  speed_threshold: float = 0.1  # m/s
  ramp_rate: float = 1.0 / 128.0  # scale per simulated second (~200 iterations)
  speed_ema_s: float = 2.0  # EMA time constant (simulated seconds)
  warmup_s: float = 0.0  # ignore envs younger than this for the speed


def curriculum_state(
  env: ManagerBasedRlEnv, cfg: PenaltyCurriculumCfg | None = None
) -> CurriculumState:
  """The env's curriculum state, created from cfg on first use."""
  state = getattr(env, "_penalty_curriculum", None)
  if state is None:
    state = CurriculumState(scale=(cfg or PenaltyCurriculumCfg()).initial_scale)
    env._penalty_curriculum = state  # type: ignore[attr-defined]
  return state


def track_speed(func: Callable[..., torch.Tensor], cfg: PenaltyCurriculumCfg) -> Callable:
  """Wrap the forward-velocity reward so it also advances the curriculum."""

  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    state = curriculum_state(env, cfg)
    active = env.episode_length_buf * env.step_dt > cfg.warmup_s
    if active.any():
      vx = env.scene["robot"].data.root_link_lin_vel_w[active, 0].mean().item()
      alpha = env.step_dt / cfg.speed_ema_s
      state.speed += alpha * (vx - state.speed)
    if state.speed > cfg.speed_threshold:
      state.scale = min(1.0, state.scale + cfg.ramp_rate * env.step_dt)
    return func(env, **kwargs)

  return wrapped


def scaled(func: Callable[..., torch.Tensor], cfg: PenaltyCurriculumCfg) -> Callable:
  """Wrap a reward term so it is multiplied by the curriculum scale."""

  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return func(env, **kwargs) * curriculum_state(env, cfg).scale

  return wrapped


def curriculum_log(
  env: ManagerBasedRlEnv, env_ids, cfg: PenaltyCurriculumCfg
) -> dict[str, float]:
  """Curriculum term (no effect): logs Curriculum/<term>/{scale,speed}."""
  state = curriculum_state(env, cfg)
  return {"scale": state.scale, "speed": state.speed}


class CurriculumPpo(PPO):
  """PPO whose symmetry mirror-loss coefficient follows the curriculum scale."""

  def update(self, *args, **kwargs):
    if self.symmetry is not None:
      if not hasattr(self, "_mirror_coeff"):
        self._mirror_coeff = self.symmetry.mirror_loss_coeff
      env = getattr(self.symmetry.env, "unwrapped", self.symmetry.env)
      self.symmetry.mirror_loss_coeff = self._mirror_coeff * curriculum_state(env).scale
    return super().update(*args, **kwargs)
