"""Microban-Crawl-Arms: the Crawl 3 gait plus arms, fine-tuned from Crawl 3.

Everything from Microban-Crawl (= Crawl 3: rewards and penalties, 0.2 m/s
speed cap, 0.06 m low-trunk threshold, flip termination, observations, PPO and
mirror symmetry) is kept unchanged, so its final checkpoint loads as-is.
Crawl 3 never puts its arms down; the added terms make the arms push in
diagonal coordination with the legs (left arm with right leg, right arm with
left leg):

- arm_contact: share of arms (forearm: hand, mid-forearm or elbow, any part)
  touching the ground. Sliding arm contacts are already penalized by the
  existing contact_slip term, which sums over every contact body incl. the
  forearms, so an arm has to push or pull instead of being dragged.
- arm_leg_diagonal: diagonal pairs (left arm + right leg, right arm + left leg)
  on the ground together. arm_leg_ipsilateral: small penalty for same-side
  pairs on the ground together.
- arm_contact_balance: |left - right| arm contact ratio over the last ~1.5 s.
- arm_pose_asym: left/right asymmetry of the mean arm joint offsets over the
  last ~1.5 s (the arms must touch down with the same part and pose).
- The arm terms start at ARM_SCALE_START of their weight and grow linearly to
  full weight over ARM_RAMP_ITERATIONS, so the Crawl 3 leg gait is not
  overwhelmed at the start of the fine-tune.
- Motor speed (same criterion as CPG run 5): the commanded joint target may
  move at most 80% of the XL330 no-load speed (MAX_TARGET_SPEED, 10.3 rad/s).
  Crawl 3 relies on target jumps of up to ~1.2 rad per step (its legs only
  reach 11 rad/s, the servo saturates): with the 10.3 rad/s limit applied at
  once its speed drops from 0.21 to 0.04 m/s (30 rad/s: 0.14, 60 rad/s: 0.21).
  So the limit starts at START_TARGET_SPEED and tightens geometrically to
  MAX_TARGET_SPEED over SPEED_RAMP_ITERATIONS; play always uses the final one.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from microban_rl.crawl_env_cfg import (
  microban_crawl_env_cfg,
  microban_crawl_ppo_runner_cfg,
)
from microban_rl.robot import XL330_NO_LOAD_SPEED
from microban_rl.symmetry import mirror_joint_map

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Ground-contact sensor bodies of each limb. The forearm (radius) carries both
# arm collision cylinders (elbow and hand); the upper arm has none.
LIMB_BODIES: dict[str, tuple[str, ...]] = {
  "arm_L": ("radius_2",),
  "arm_R": ("radius",),
  "leg_L": (
    "hip_block__configuration_left",
    "femur__configuration_left",
    "tibia__configuration_left",
    "ankle_block__configuration_right",
    "foot_2",
  ),
  "leg_R": (
    "hip_block__configuration_right",
    "femur__configuration_right",
    "tibia__configuration_right",
    "ankle_block__configuration_left",
    "foot",
  ),
}
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
ARM_TERMS = (
  "arm_contact",
  "arm_leg_diagonal",
  "arm_leg_ipsilateral",
  "arm_contact_balance",
  "arm_pose_asym",
)
BALANCE_WINDOW_S = 1.5  # EMA time constant for the balance terms
ARM_SCALE_START = 0.1
ARM_RAMP_ITERATIONS = 500
STEPS_PER_ITERATION = 24  # == microban_crawl_ppo_runner_cfg().num_steps_per_env
POLICY_DT = 0.02
MAX_TARGET_SPEED = 0.8 * XL330_NO_LOAD_SPEED  # 10.3 rad/s, as in cpg.py
START_TARGET_SPEED = 60.0  # rad/s, where Crawl 3 keeps its full speed
SPEED_RAMP_ITERATIONS = 400


def steps_since_start(env: ManagerBasedRlEnv) -> int:
  """Env steps since this run's first step.

  The runner restores common_step_counter from the checkpoint (Crawl 3 ends at
  36000), so the ramps count from the value seen at the first env step.
  Only called from step-time code (actions, rewards), never at env init.
  """
  base = env.__dict__.setdefault("_ramp_base_step", env.common_step_counter)
  return env.common_step_counter - base


##
# Speed-limited joint position action.
##


@dataclass(kw_only=True)
class SpeedLimitedJointPositionActionCfg(JointPositionActionCfg):
  """JointPositionActionCfg whose target moves at most max_speed rad/s.

  The limit goes from start_speed to max_speed (geometrically) over the first
  ramp_steps env steps; ramp_steps 0 = max_speed from the start.
  """

  max_speed: float = MAX_TARGET_SPEED
  start_speed: float = START_TARGET_SPEED
  ramp_steps: int = 0

  def build(self, env: ManagerBasedRlEnv) -> SpeedLimitedJointPositionAction:
    return SpeedLimitedJointPositionAction(self, env)


class SpeedLimitedJointPositionAction(JointPositionAction):
  """Clamps each policy step's target change to max_speed x policy dt.

  Starts from the measured joint positions after a reset.
  """

  cfg: SpeedLimitedJointPositionActionCfg

  def __init__(self, cfg: SpeedLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._prev = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def speed_limit(self) -> float:
    cfg = self.cfg
    if cfg.ramp_steps <= 0:
      return cfg.max_speed
    t = min(1.0, steps_since_start(self._env) / cfg.ramp_steps)
    return cfg.start_speed * (cfg.max_speed / cfg.start_speed) ** t

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    max_step = self.speed_limit * self._env.step_dt
    if self._fresh.any():
      q = self._entity.data.joint_pos[:, self._target_ids]
      self._prev[self._fresh] = q[self._fresh]
      self._fresh[:] = False
    self._processed_actions = torch.clamp(
      self._processed_actions, self._prev - max_step, self._prev + max_step
    )
    self._prev = self._processed_actions.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._fresh[slice(None) if env_ids is None else env_ids] = True


##
# Arm rewards.
##


def limb_contacts(
  env: ManagerBasedRlEnv, sensor_name: str = "ground_contact"
) -> dict[str, torch.Tensor]:
  """Per limb: any of its bodies touching the ground. {name: bool [B]}"""
  sensor = env.scene[sensor_name]
  cache = getattr(env, "_limb_contact_ids", None)
  if cache is None:
    names = list(sensor.primary_names)
    cache = {
      k: torch.tensor([names.index(b) for b in v], device=env.device)
      for k, v in LIMB_BODIES.items()
    }
    env._limb_contact_ids = cache  # type: ignore[attr-defined]
  found = sensor.data.found > 0
  return {k: found[:, ids].any(dim=1) for k, ids in cache.items()}


def arm_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  c = limb_contacts(env)
  return 0.5 * (c["arm_L"].float() + c["arm_R"].float())


def arm_leg_diagonal(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Left arm + right leg, right arm + left leg on the ground together."""
  c = limb_contacts(env)
  return 0.5 * ((c["arm_L"] & c["leg_R"]).float() + (c["arm_R"] & c["leg_L"]).float())


def arm_leg_ipsilateral(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Left arm + left leg, right arm + right leg on the ground together."""
  c = limb_contacts(env)
  return 0.5 * ((c["arm_L"] & c["leg_L"]).float() + (c["arm_R"] & c["leg_R"]).float())


def _ema(env: ManagerBasedRlEnv, key: str, x: torch.Tensor, tau_s: float) -> torch.Tensor:
  """Per-env exponential moving average, restarted at the episode start."""
  store = env.__dict__.setdefault("_arm_ema", {})
  if key not in store:
    store[key] = x.clone()
  ema = store[key]
  fresh = env.episode_length_buf <= 1
  ema[fresh] = x[fresh]
  ema += (env.step_dt / tau_s) * (x - ema)
  return ema


def arm_contact_balance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """|left - right| arm ground-contact ratio over the last ~BALANCE_WINDOW_S."""
  c = limb_contacts(env)
  x = torch.stack([c["arm_L"], c["arm_R"]], dim=1).float()
  ema = _ema(env, "arm_contact", x, BALANCE_WINDOW_S)
  return torch.abs(ema[:, 0] - ema[:, 1])


def arm_pose_asym(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean |q_L - sign q_R| of the arm joints' offsets over ~BALANCE_WINDOW_S."""
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  cache = getattr(env, "_arm_pose_ids", None)
  if cache is None:
    idx, sign = mirror_joint_map(names, env.device)
    left = torch.tensor([names.index(f"left_{j}") for j in ARM_JOINTS], device=env.device)
    cache = (left, idx[left], sign[left])
    env._arm_pose_ids = cache  # type: ignore[attr-defined]
  left, right, sign = cache
  q = robot.data.joint_pos - robot.data.default_joint_pos
  ema = _ema(env, "arm_pose", torch.cat([q[:, left], q[:, right]], dim=1), BALANCE_WINDOW_S)
  k = len(ARM_JOINTS)
  return torch.mean(torch.abs(ema[:, :k] - sign * ema[:, k:]), dim=1)


def arm_scale(env: ManagerBasedRlEnv) -> float:
  """ARM_SCALE_START -> 1 linearly over ARM_RAMP_ITERATIONS."""
  t = steps_since_start(env) / (ARM_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  return min(1.0, ARM_SCALE_START + (1.0 - ARM_SCALE_START) * t)


def arm_ramped(func: Callable[..., torch.Tensor]) -> Callable:
  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return func(env, **kwargs) * arm_scale(env)

  return wrapped


def arm_scale_log(env: ManagerBasedRlEnv, env_ids) -> dict[str, float]:
  """Curriculum term (no effect): logs Curriculum/arms/{scale,speed_limit}."""
  if not hasattr(env, "_ramp_base_step"):  # reset at env init, before the checkpoint load
    return {"scale": ARM_SCALE_START, "speed_limit": START_TARGET_SPEED}
  action = env.action_manager.get_term("joint_pos")
  return {"scale": arm_scale(env), "speed_limit": action.speed_limit}


##
# Environment.
##


def microban_crawl_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = microban_crawl_env_cfg(play=play)
  old = cfg.actions["joint_pos"]
  cfg.actions = {
    "joint_pos": SpeedLimitedJointPositionActionCfg(
      entity_name=old.entity_name,
      actuator_names=old.actuator_names,
      scale=old.scale,
      use_default_offset=old.use_default_offset,
      ramp_steps=0 if play else SPEED_RAMP_ITERATIONS * STEPS_PER_ITERATION,
    )
  }
  cfg.rewards.update({
    "arm_contact": RewardTermCfg(func=arm_contact, weight=0.3),
    "arm_leg_diagonal": RewardTermCfg(func=arm_leg_diagonal, weight=1.0),
    "arm_leg_ipsilateral": RewardTermCfg(func=arm_leg_ipsilateral, weight=-0.5),
    "arm_contact_balance": RewardTermCfg(func=arm_contact_balance, weight=-1.0),
    "arm_pose_asym": RewardTermCfg(func=arm_pose_asym, weight=-1.0),
  })
  for name in ARM_TERMS:
    cfg.rewards[name].func = arm_ramped(cfg.rewards[name].func)
  cfg.curriculum = {"arms": CurriculumTermCfg(func=arm_scale_log)}
  return cfg


def microban_crawl_arms_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = microban_crawl_ppo_runner_cfg()
  cfg.experiment_name = "microban_crawl_arms"
  cfg.max_iterations = 2000  # on top of Crawl 3's 1500
  return cfg


##
# Microban-Crawl-Arms-2: fine-tune of Crawl-Arms, whose arms never touched down.
##
# Crawl-Arms kept its forearms ~27 mm above the floor (p5 20 mm): the contact
# rewards never fired, and the only arm term with a signal, arm_pose_asym,
# penalized exploring arm motions. Changes:
# - arm_floor_guide: dense reward for each forearm's lowest point being close to
#   the floor, larger for the arm whose diagonal leg is on the ground.
# - arm_pose_asym switches on (ramped) only once the batch-mean ground contact
#   of each arm exceeds POSE_GATE_CONTACT.
# - Larger exploration noise on the arm joints at the start (ArmNoiseRunner).
# - The motor speed limit is the final 10.3 rad/s from the start (Crawl-Arms
#   already adapted to it); the existing arm terms keep their full weight and
#   only the new guide reward is ramped in.

GUIDE_HEIGHT_SCALE = 0.02  # m, reward exp(-h / scale)
GUIDE_BASE = 0.25  # weight of an arm whose diagonal leg is in the air (1 if down)
POSE_GATE_CONTACT = 0.10
POSE_GATE_EMA_S = 2.0
POSE_GATE_RAMP_ITERATIONS = 100
ARM_NOISE_STD = 0.4
ARM_ACTION_JOINTS = tuple(f"{s}_{j}" for s in ("left", "right") for j in ARM_JOINTS)


def forearm_heights(env: ManagerBasedRlEnv, sensor_name: str = "ground_contact") -> torch.Tensor:
  """Height of the lowest forearm collision point above the floor. [B, 2] (L, R)"""
  from microban_rl.crawl_env_cfg import _collision_points
  from mjlab.utils.lab_api.math import quat_apply

  robot = env.scene["robot"]
  sensor = env.scene[sensor_name]
  prim, body, local, radius = _collision_points(env, robot, sensor)
  names = list(sensor.primary_names)
  pos = robot.data.body_link_pos_w[:, body]
  quat = robot.data.body_link_quat_w[:, body]
  height = pos[..., 2] + quat_apply(quat, local.expand_as(pos))[..., 2] - radius
  out = []
  for arm in ("radius_2", "radius"):
    mask = prim == names.index(arm)
    out.append(height[:, mask].amin(dim=1))
  return torch.stack(out, dim=1)


def arm_floor_guide(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean over arms of exp(-h / scale), weighted up when the diagonal leg is down."""
  c = limb_contacts(env)
  prox = torch.exp(-forearm_heights(env).clamp(min=0.0) / GUIDE_HEIGHT_SCALE)
  diag = torch.stack([c["leg_R"], c["leg_L"]], dim=1).float()  # for arm L, arm R
  w = GUIDE_BASE + (1.0 - GUIDE_BASE) * diag
  return torch.mean(w * prox, dim=1)


def pose_gate(env: ManagerBasedRlEnv) -> float:
  """0 until both arms' batch-mean contact ratios pass POSE_GATE_CONTACT, then ramps to 1."""
  st = env.__dict__.setdefault("_pose_gate", {"ema": torch.zeros(2, device=env.device), "on": None})
  c = limb_contacts(env)
  x = torch.stack([c["arm_L"], c["arm_R"]]).float().mean(dim=1)
  st["ema"] += (env.step_dt / POSE_GATE_EMA_S) * (x - st["ema"])
  step = steps_since_start(env)
  if st["on"] is None and bool((st["ema"] > POSE_GATE_CONTACT).all()):
    st["on"] = step
  if st["on"] is None:
    return 0.0
  return min(1.0, (step - st["on"]) / (POSE_GATE_RAMP_ITERATIONS * STEPS_PER_ITERATION))


def gated_pose_asym(env: ManagerBasedRlEnv) -> torch.Tensor:
  return arm_pose_asym(env) * pose_gate(env)


def arms2_log(env: ManagerBasedRlEnv, env_ids) -> dict[str, float]:
  """Curriculum term (no effect): logs the guide ramp, pose gate and arm contact EMAs."""
  if not hasattr(env, "_ramp_base_step"):
    return {"guide_scale": ARM_SCALE_START, "pose_gate": 0.0}
  st = env.__dict__.get("_pose_gate")
  ema = st["ema"].tolist() if st else [0.0, 0.0]
  on = st is not None and st["on"] is not None
  gate = 0.0 if not on else min(
    1.0, (steps_since_start(env) - st["on"]) / (POSE_GATE_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  )
  return {"guide_scale": arm_scale(env), "pose_gate": gate,
          "arm_contact_L": ema[0], "arm_contact_R": ema[1]}


def microban_crawl_arms2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = microban_crawl_arms_env_cfg(play=play)
  cfg.actions["joint_pos"].ramp_steps = 0
  for name in ARM_TERMS:  # full weight, as at the end of Crawl-Arms
    cfg.rewards[name].func = cfg.rewards[name].func.__wrapped__
  cfg.rewards["arm_pose_asym"].func = gated_pose_asym
  cfg.rewards["arm_floor_guide"] = RewardTermCfg(
    func=arm_ramped(arm_floor_guide), weight=0.5
  )
  cfg.curriculum = {"arms": CurriculumTermCfg(func=arms2_log)}
  return cfg


def microban_crawl_arms2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = microban_crawl_arms_ppo_runner_cfg()  # same experiment: resumes Crawl-Arms runs
  cfg.max_iterations = 2000
  return cfg


from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402


class ArmNoiseRunner(MjlabOnPolicyRunner):
  """Raises the arm joints' action std to at least ARM_NOISE_STD when training starts."""

  def learn(self, *args, **kwargs):
    names = list(self.env.unwrapped.action_manager.get_term("joint_pos").target_names)
    std = self.alg.actor.distribution.std_param
    with torch.no_grad():
      for n in ARM_ACTION_JOINTS:
        i = names.index(n)
        std[i] = max(float(std[i]), ARM_NOISE_STD)
    print(f"[ArmNoiseRunner] action std: {[round(float(s), 3) for s in std]}", flush=True)
    return super().learn(*args, **kwargs)
