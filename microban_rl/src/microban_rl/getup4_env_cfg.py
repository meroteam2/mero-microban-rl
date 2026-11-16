"""Microban-GetUp-4: Microban-GetUp-3 plus penalties against the one-arm, twisted push.

Run 3's model_1500 (logs/rsl_rl/microban_getup/2026-11-16_20-30-14_getup3)
gets up in 100% of the envs, but after pike it holds itself on the left arm
only and the whole trunk twists sideways (the joint-angle asymmetry metric
does not see it). Changes, otherwise identical to Microban-GetUp-3, all inside
the push window (pike_1 -> squat_deep):

1. trunk_twist: (trunk lateral roll)^2 + (trunk yaw change since pike_1)^2, weight -3.
   Lateral roll = asin(world z of the trunk's left (+y) axis), valid at any
   forward tilt; yaw = heading of the trunk's +y axis on the ground.
2. one_hand: -2 per step while exactly one hand touches the floor (both hands
   should leave the floor at about the same time).
3. hand_load_diff: |F_L - F_R| / robot weight (normal forces of the forearms
   on the floor) while a hand is down, weight -0.5.

Resumes run 3's model_1500; the regularization ramp continues from its value
there (0.7) and reaches 1 at iteration 2000 as before.
"""

from __future__ import annotations

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply

from microban_rl.crawl_env_cfg import robot_weight
from microban_rl.getup3_env_cfg import (
  REG3_RAMP_ITERATIONS,
  REG3_START,
  microban_getup3_env_cfg,
  microban_getup3_ppo_runner_cfg,
)
from microban_rl.getup_env_cfg import (
  GETUP_CONTACT_SENSOR,
  STEPS_PER_ITERATION,
  elapsed_steps,
  in_window,
  ref_time,
  reference,
  steps_since_start,
)

HAND_FORCE_SENSOR = ContactSensorCfg(
  name="hand_force",
  primary=ContactMatch(mode="body", pattern=r"^(radius|radius_2)$", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
)
# Run 3's ramp at its iteration 1500 (500 iterations after its start).
REG4_START = REG3_START + (1.0 - REG3_START) * 500 / REG3_RAMP_ITERATIONS
REG4_RAMP_ITERATIONS = REG3_RAMP_ITERATIONS - 500


def push_metrics(env) -> dict[str, torch.Tensor]:
  """Per env, this step: lateral roll, yaw change since pike_1, hands down (L, R), hand normal forces (L, R)."""
  key = env._sim_step_counter
  cache = env.__dict__.get("_push_metrics")
  if cache is not None and cache[0] == key:
    return cache[1]
  robot = env.scene["robot"]
  B = env.num_envs
  ey = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(B, 3)
  left = quat_apply(robot.data.root_link_quat_w, ey)
  lat = torch.asin(left[:, 2].clamp(-1, 1))
  yaw = torch.atan2(left[:, 1], left[:, 0])
  yaw0 = env.__dict__.get("_push_yaw0")
  if yaw0 is None:
    yaw0 = env._push_yaw0 = yaw.clone()  # type: ignore[attr-defined]
  # Track the heading until the push window starts (or at an episode start), then hold it.
  before = (ref_time(env) < reference(env).kf_t["pike_1"]) | (elapsed_steps(env) <= 1)
  yaw0[:] = torch.where(before, yaw, yaw0)
  dyaw = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
  cs = env.scene[GETUP_CONTACT_SENSOR.name]
  pn = list(cs.primary_names)
  found = cs.data.found > 0
  hands = torch.stack([found[:, pn.index("radius_2")], found[:, pn.index("radius")]], 1)  # L, R
  fs = env.scene[HAND_FORCE_SENSOR.name]
  fn = list(fs.primary_names)
  f = fs.data.force[..., 2].abs()  # [B, 2] (sensor order)
  force = torch.stack([f[:, fn.index("radius_2")], f[:, fn.index("radius")]], 1)
  out = {"lat": lat, "dyaw": dyaw, "hands": hands, "force": force}
  env._push_metrics = (key, out)  # type: ignore[attr-defined]
  return out


def trunk_twist(env) -> torch.Tensor:
  m = push_metrics(env)
  return (m["lat"].square() + m["dyaw"].square()) * in_window(env).float()


def one_hand(env) -> torch.Tensor:
  m = push_metrics(env)
  return ((m["hands"].sum(1) == 1) & in_window(env)).float()


def hand_load_diff(env) -> torch.Tensor:
  m = push_metrics(env)
  any_down = m["hands"].any(1) & in_window(env)
  return (m["force"][:, 0] - m["force"][:, 1]).abs() / robot_weight(env) * any_down.float()


def reg4_scale(env) -> float:
  t = steps_since_start(env) / (REG4_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  return min(1.0, REG4_START + (1.0 - REG4_START) * t)


def microban_getup4_env_cfg(play: bool = False):
  cfg = microban_getup3_env_cfg(play=play)
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (HAND_FORCE_SENSOR,)
  for name in ("torques", "action_rate", "joint_limits", "contact_slip"):
    base = cfg.rewards[name].func.__defaults__[0]  # the unramped term

    def ramped(env, _f=base, **kw):
      return _f(env, **kw) * reg4_scale(env)

    ramped.__name__ = f"{base.__name__}_reg4"
    cfg.rewards[name].func = ramped
  cfg.rewards["trunk_twist"] = RewardTermCfg(func=trunk_twist, weight=-3.0)
  cfg.rewards["one_hand"] = RewardTermCfg(func=one_hand, weight=-2.0)
  cfg.rewards["hand_load_diff"] = RewardTermCfg(func=hand_load_diff, weight=-0.5)
  return cfg


def microban_getup4_ppo_runner_cfg():
  cfg = microban_getup3_ppo_runner_cfg()
  cfg.max_iterations = 2500  # 1500 -> 4000
  cfg.run_name = "getup4"
  return cfg
