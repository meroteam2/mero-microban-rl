"""Microban-Crawl-CPG: the Microban-Crawl task driven by a CPG action.

Everything from Microban-Crawl is kept (prone start, rewards incl. contact slip
and contact force penalties, 0.2 m/s speed cap, low-trunk threshold, flip
termination, left/right mirror symmetry). Changes:

- Action: CPG parameters + joint residuals (cpg.py) instead of direct joint
  position offsets; the CPG phases are observed.
- Rewards are zeroed for the first REWARD_WARMUP_S seconds of every episode
  (terminations still apply), so violent start-up motions earn nothing.
- Longer episodes (20 s) for gamma = 0.999.
- PPO: gamma 0.999, lambda 0.97, unclipped value loss, 8 epochs, 3000
  iterations, temporally correlated (OU) exploration noise (noise.py).
- Actions clipped to [-2, 2] before the env, std capped at 1.0 and a smaller
  entropy bonus. Without these the first run diverged: the 57-dim entropy
  bonus grew the std to ~3, the raw (unclipped) actions drifted in the CPG's
  saturated regions, and the action-rate penalty on them exploded.
- Penalty curriculum (curriculum.py): contact slip, contact force, touchdown
  speed, action rate, the low-trunk reward and the symmetry mirror loss start
  at 0 and fade in to their full weight once the mean forward speed exceeds
  0.1 m/s. The second run without it never moved: with exploration noise these
  terms cost more than the (still ~0) forward reward, the free low-trunk reward
  made lying still worth +0.9/s, and the policy pushed its CPG parameters into
  the action clip, where the noise (and so the action-rate penalty) vanishes.
- Left/right gait symmetry (the fourth run drifted into an asymmetric gait that
  the mirror loss allows, since the mirrored gait is equally valid): the CPG
  phase difference is limited to pi +- 0.2 rad (cpg.py), a half-cycle mirror
  reward pays for the joint state matching the mirror of the state half a CPG
  period earlier, and a small penalty keeps the head facing forward.
- Arms push with the hand: ground contact of the elbow cylinder of the forearm
  is penalized (the forearm has two collision cylinders, elbow and hand).
- Motor speed: robot.py's servo model already saturates at the XL330 no-load
  speed (see XL330_NO_LOAD_SPEED); the CPG amplitude is capped so that the
  commanded peak speed amplitude * 2 pi f stays below 80% of it (cpg.py).
"""

from __future__ import annotations

import functools
from typing import Callable

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from microban_rl.cpg import CpgAction, CpgActionCfg, cpg_phase
from microban_rl.crawl_env_cfg import microban_crawl_env_cfg
from microban_rl.curriculum import (
  PenaltyCurriculumCfg,
  curriculum_log,
  scaled,
  track_speed,
)
from microban_rl.prone import get_microban_prone_robot_cfg
from microban_rl.robot import get_spec
from microban_rl.symmetry import SymmetricPpoAlgorithmCfg, mirror_joint_map

REWARD_WARMUP_S = 2.0
EPISODE_LENGTH_S = 20.0
POLICY_DT = 0.02  # decimation 4 x 5 ms
NOISE_CORRELATION_S = 0.25
# Reward terms faded in by the penalty curriculum (plus the PPO mirror loss).
CURRICULUM_TERMS = (
  "contact_slip",
  "contact_force",
  "touchdown_speed",
  "action_rate",
  "trunk_low",
  "half_cycle_mirror",
  "elbow_contact",
)
HALF_CYCLE_SIGMA = 0.3  # rad, RMS joint mismatch at which the reward is 1/e
_HISTORY = 40  # policy steps kept; half a period is at most 0.5 / 0.8 Hz = 32 steps


##
# Arm collision geoms (the MJCF leaves them unnamed).
##


def get_spec_with_arm_geom_names() -> mujoco.MjSpec:
  """robot.get_spec() with the two forearm capsules named per side.

  radius / radius_2 each have a large collision cylinder around the elbow
  servo (r 9 mm) and a small one at the forearm tip (r 5.2 mm), the "hand".
  """
  spec = get_spec()
  for body_name, side in (("radius", "right"), ("radius_2", "left")):
    cylinders = [g for g in spec.body(body_name).geoms
                 if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER and g.contype]
    assert len(cylinders) == 2, body_name
    for g in cylinders:
      part = "elbow" if g.size[0] > 0.007 else "hand"
      g.name = f"{side}_{part}_collision"
  return spec


ARM_CONTACT_SENSOR = ContactSensorCfg(
  name="arm_contact",
  primary=ContactMatch(
    mode="geom",
    pattern=(r"^(left|right)_(elbow|hand)_collision$",),
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="mindist",
  num_slots=1,
)


##
# Rewards.
##


def elbow_contact(env: ManagerBasedRlEnv, sensor_name: str = "arm_contact") -> torch.Tensor:
  """Number of elbow cylinders (0-2) touching the ground."""
  sensor = env.scene[sensor_name]
  elbow = torch.tensor(["elbow" in n for n in sensor.primary_names], device=env.device)
  return ((sensor.data.found > 0) & elbow).float().sum(dim=1)


def head_pos_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared head yaw angle (0 = facing forward)."""
  robot = env.scene["robot"]
  i = robot.joint_names.index("head")
  return torch.square(robot.data.joint_pos[:, i] - robot.data.default_joint_pos[:, i])


def half_cycle_mirror(
  env: ManagerBasedRlEnv, sigma: float, action_name: str = "cpg"
) -> torch.Tensor:
  """exp(-mean_j (q_j(t) - mirror(q(t - T/2))_j)^2 / sigma^2), T = CPG period.

  A left/right symmetric alternating gait repeats mirrored every half period,
  so this is 1 for it. Joint offsets from the (symmetric) default pose; the
  half-period delay is interpolated between policy steps. The history spans
  episode resets, which the reward warm-up (2 s > max delay 0.625 s) covers.
  """
  robot = env.scene["robot"]
  q = robot.data.joint_pos - robot.data.default_joint_pos
  state = getattr(env, "_half_cycle_history", None)
  if state is None:
    idx, sign = mirror_joint_map(list(robot.joint_names), env.device)
    buf = q.unsqueeze(1).repeat(1, _HISTORY, 1)
    state = {"buf": buf, "ptr": 0, "idx": idx, "sign": sign}
    env._half_cycle_history = state  # type: ignore[attr-defined]
  buf = state["buf"]
  state["ptr"] = (state["ptr"] + 1) % _HISTORY
  buf[:, state["ptr"]] = q
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, CpgAction)
  lag = (0.5 / (term.frequency * env.step_dt)).clamp(1.0, _HISTORY - 2.0)
  lo = lag.floor().long()
  w = (lag - lo)[:, None]
  rows = torch.arange(q.shape[0], device=env.device)
  q_lo = buf[rows, (state["ptr"] - lo) % _HISTORY]
  q_hi = buf[rows, (state["ptr"] - lo - 1) % _HISTORY]
  past = (1 - w) * q_lo + w * q_hi
  mirrored = past[:, state["idx"]] * state["sign"]
  err = torch.mean(torch.square(q - mirrored), dim=1)
  return torch.exp(-err / sigma**2)


def after_warmup(func: Callable[..., torch.Tensor], warmup_s: float) -> Callable:
  """Wrap a reward term so it returns 0 during the first warmup_s of an episode."""

  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    active = env.episode_length_buf * env.step_dt > warmup_s
    return func(env, **kwargs) * active

  return wrapped


def microban_crawl_cpg_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = microban_crawl_env_cfg(play=play)

  cfg.actions = {"cpg": CpgActionCfg(entity_name="robot")}
  cfg.scene.entities["robot"] = get_microban_prone_robot_cfg(
    spec_fn=get_spec_with_arm_geom_names
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (ARM_CONTACT_SENSOR,)

  cfg.rewards["half_cycle_mirror"] = RewardTermCfg(
    func=half_cycle_mirror, weight=1.0, params={"sigma": HALF_CYCLE_SIGMA}
  )
  cfg.rewards["head_pos"] = RewardTermCfg(func=head_pos_l2, weight=-0.5)
  cfg.rewards["elbow_contact"] = RewardTermCfg(func=elbow_contact, weight=-1.0)

  phase = ObservationTermCfg(func=cpg_phase, params={"action_name": "cpg"})
  for group in ("actor", "critic"):
    cfg.observations[group].terms["cpg_phase"] = phase

  # play: full penalties from the start (only matters for diagnostics).
  curriculum = PenaltyCurriculumCfg(
    initial_scale=1.0 if play else 0.0, warmup_s=REWARD_WARMUP_S
  )
  cfg.rewards["forward_vel"].func = track_speed(cfg.rewards["forward_vel"].func, curriculum)
  for name in CURRICULUM_TERMS:
    cfg.rewards[name].func = scaled(cfg.rewards[name].func, curriculum)
  for term in cfg.rewards.values():
    term.func = after_warmup(term.func, REWARD_WARMUP_S)
  cfg.curriculum = {
    "penalties": CurriculumTermCfg(func=curriculum_log, params={"cfg": curriculum})
  }

  if not play:
    cfg.episode_length_s = EPISODE_LENGTH_S
  return cfg


def microban_crawl_cpg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  noise = {
    "class_name": "microban_rl.noise:ColoredGaussianDistribution",
    # Start inside std_range: rsl-rl clamps std_param, so a std that starts at
    # the upper bound and is nudged above it by the entropy bonus gets zero
    # gradient and stays frozen at 1.0 (as in the second run).
    "init_std": 0.5,
    "std_range": (0.05, 1.0),
    "std_type": "scalar",
    "correlation_time_s": NOISE_CORRELATION_S,
    "step_dt": POLICY_DT,
  }
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=noise,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=SymmetricPpoAlgorithmCfg(
      class_name="microban_rl.curriculum:CurriculumPpo",
      value_loss_coef=1.0,
      # Returns grow ~10x with gamma 0.999; a +-0.2 value clip would make the
      # critic crawl toward them.
      use_clipped_value_loss=False,
      clip_param=0.2,
      # Entropy is summed over 57 action dims (19 before): scale per dim to match.
      entropy_coef=0.002,
      num_learning_epochs=8,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.999,
      lam=0.97,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    clip_actions=2.0,
    experiment_name="microban_crawl_cpg",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=3000,
  )
