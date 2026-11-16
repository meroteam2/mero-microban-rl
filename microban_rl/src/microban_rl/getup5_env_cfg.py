"""Microban-GetUp-5: Microban-GetUp-4 plus a stand-pitch centering reward.

Run 4's model_2000 (logs/rsl_rl/microban_getup/2026-11-16_20-47-08_getup4)
pushes with both hands symmetrically (trunk roll 3 deg, yaw 8 deg) but ends
standing at pitch +0.200 rad, the upper edge of the approved stand range
(0 .. +0.2), so the 1 s stand hold rarely completes. Changes, otherwise
identical to Microban-GetUp-4:

1. stand_pitch: exp(-(pitch - 0.175)^2 / 0.05^2) from the crouch keyframe
   time (11.4 s) on, weight 1 (0.175 = the stand_target keyframe's pitch).
2. Regularization at full weight (run 4's ramp reached 1 at iteration 2000).
3. FOOT_FORCE_SENSOR (sensing only): per-substep normal force of each foot,
   for the landing-impact report.

Resumes run 4's model_2000.
"""

from __future__ import annotations

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from microban_rl.getup4_env_cfg import microban_getup4_env_cfg, microban_getup4_ppo_runner_cfg
from microban_rl.getup_env_cfg import getup_state, ref_time, reference

STAND_PITCH_TARGET = 0.175
STAND_PITCH_SIGMA = 0.05

FOOT_FORCE_SENSOR = ContactSensorCfg(
  name="foot_force",
  primary=ContactMatch(mode="body", pattern=r"^(foot|foot_2)$", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  history_length=4,  # == decimation: every physics substep of the last policy step
)


def stand_pitch(env) -> torch.Tensor:
  st = getup_state(env)
  late = ref_time(env) >= reference(env).kf_t["crouch"]
  return torch.exp(-(st["pitch"] - STAND_PITCH_TARGET).square() / STAND_PITCH_SIGMA**2) * late.float()


def microban_getup5_env_cfg(play: bool = False):
  cfg = microban_getup4_env_cfg(play=play)
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (FOOT_FORCE_SENSOR,)
  for name in ("torques", "action_rate", "joint_limits", "contact_slip"):
    cfg.rewards[name].func = cfg.rewards[name].func.__defaults__[0]  # unramped (full weight)
  cfg.rewards["stand_pitch"] = RewardTermCfg(func=stand_pitch, weight=1.0)
  return cfg


def microban_getup5_ppo_runner_cfg():
  cfg = microban_getup4_ppo_runner_cfg()
  cfg.max_iterations = 2000  # 2000 -> 4000
  cfg.run_name = "getup5"
  return cfg
