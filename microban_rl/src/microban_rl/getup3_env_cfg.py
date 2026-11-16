"""Microban-GetUp-3: Microban-GetUp-2 plus pressure to leave the hands-down pike.

Run 2 (logs/rsl_rl/microban_getup/2026-11-16_20-08-04_getup2) fixed the
one-sided twist (push-segment hip_yaw left/right difference 0.82 -> 0.20 rad)
but settled in pike_push: at iteration 1500, 93% of the envs stayed there with
the hands down. Changes, otherwise identical to Microban-GetUp-2:

1. hands_after_fold: -1 per step while a hand touches the floor after the
   fold keyframe time (9.4 s).
2. weight_transfer weight 2 -> 4.
3. Reference-state starts: fold picked 4x as often as pike_1 / pike_push /
   crouch (was 2x), squat_deep 2x.
4. The x3 imitation tolerance applies from pike_1 only until fold (9.4 s),
   not until squat_deep: staying in pike_push after the reference moved on
   costs the full imitation reward. (Twist limit and push_symmetry keep the
   pike_1 -> squat_deep window.)
5. Resumes run 2's model_1000. The regularization ramp continues from where
   model_1000 was (scale 0.4) and reaches 1 at the same iteration as in
   run 2 (2000).
"""

from __future__ import annotations

import torch

from mjlab.managers.reward_manager import RewardTermCfg

from microban_rl.getup2_env_cfg import (
  REG2_RAMP_ITERATIONS,
  RSI2_KEYFRAME,
  microban_getup2_env_cfg,
  microban_getup2_ppo_runner_cfg,
)
from microban_rl.getup_env_cfg import (
  HEIGHT_SIGMA,
  POSE_SIGMA,
  REG_START,
  RSI_PRONE,
  STEPS_PER_ITERATION,
  TILT_SIGMA,
  WIDE,
  getup_state,
  ref_time,
  reference,
  steps_since_start,
)

WIDE_UNTIL = "fold"
RSI3_KEYFRAME_WEIGHTS = {"pike_1": 1.0, "pike_push": 1.0, "fold": 4.0, "squat_deep": 2.0, "crouch": 1.0}
# Run 2's ramp at its iteration 1000 (500 iterations after its start).
REG3_START = REG_START + (1.0 - REG_START) * 500 / REG2_RAMP_ITERATIONS
REG3_RAMP_ITERATIONS = REG2_RAMP_ITERATIONS - 500


def in_wide_window(env) -> torch.Tensor:
  ref = reference(env)
  t = ref_time(env)
  return (t >= ref.kf_t["pike_1"]) & (t <= ref.kf_t[WIDE_UNTIL])


def _wide3(env, base: float) -> torch.Tensor:
  return torch.where(in_wide_window(env), base * WIDE, base)


def pose_imitation3(env) -> torch.Tensor:
  ref = reference(env)
  q = env.scene["robot"].data.joint_pos
  err = (q - ref.q[ref.index(ref_time(env))]).square().mean(1)
  return torch.exp(-err / _wide3(env, POSE_SIGMA) ** 2)


def trunk_imitation3(env) -> torch.Tensor:
  ref = reference(env)
  st = getup_state(env)
  i = ref.index(ref_time(env))
  a = torch.exp(-(st["tilt"] - ref.tilt[i]).square() / _wide3(env, TILT_SIGMA) ** 2)
  b = torch.exp(-(st["z"] - ref.z[i]).square() / _wide3(env, HEIGHT_SIGMA) ** 2)
  return 0.5 * (a + b)


def hands_after_fold(env) -> torch.Tensor:
  st = getup_state(env)
  return (st["hands_down"] & (ref_time(env) >= reference(env).kf_t["fold"])).float()


def reg3_scale(env) -> float:
  t = steps_since_start(env) / (REG3_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  return min(1.0, REG3_START + (1.0 - REG3_START) * t)


def microban_getup3_env_cfg(play: bool = False):
  cfg = microban_getup2_env_cfg(play=play)
  if not play:
    ev = cfg.events["reset_getup"].params
    ev["p_prone"], ev["p_replay"] = RSI_PRONE, 1.0 - RSI_PRONE - RSI2_KEYFRAME
    ev["keyframe_weights"] = dict(RSI3_KEYFRAME_WEIGHTS)
  cfg.rewards["pose_imitation"].func = pose_imitation3
  cfg.rewards["trunk_imitation"].func = trunk_imitation3
  cfg.rewards["weight_transfer"].weight = 4.0
  cfg.rewards["hands_after_fold"] = RewardTermCfg(func=hands_after_fold, weight=-1.0)
  for name in ("torques", "action_rate", "joint_limits", "contact_slip"):
    base = cfg.rewards[name].func.__defaults__[0]  # the unramped term (see getup2 'ramped')

    def ramped(env, _f=base, **kw):
      return _f(env, **kw) * reg3_scale(env)

    ramped.__name__ = f"{base.__name__}_reg3"
    cfg.rewards[name].func = ramped
  return cfg


def microban_getup3_ppo_runner_cfg():
  cfg = microban_getup2_ppo_runner_cfg()
  cfg.max_iterations = 3000  # 1000 -> 4000
  cfg.run_name = "getup3"
  return cfg
