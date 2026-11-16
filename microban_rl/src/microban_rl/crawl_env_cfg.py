"""Microban "crawl forward while lying prone" task.

The robot starts face-down (see prone.py) and is rewarded for moving its trunk
along world +x while staying low, keeping its heading and moving smoothly.
Pure PPO, no reference motion. Contacts are penalized for sliding along the
ground, and the policy is trained left/right symmetric (symmetry.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  is_terminated,
  joint_pos_limits,
  joint_pos_rel,
  joint_torques_l2,
  joint_vel_rel,
  last_action,
  projected_gravity,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from microban_rl.prone import (
  belly_up,
  get_microban_prone_robot_cfg,
  head_direction_w,
  reset_prone,
)
from microban_rl.stand_env_cfg import STAND_HEIGHT, root_height
from microban_rl.symmetry import SymmetricPpoAlgorithmCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")

# Trunk counts as "low" below this height. Lying flat the trunk origin rests at
# 0.045 m; 0.06 m allows ~1.5 cm of lift (half the standing height would be
# STAND_HEIGHT / 2 = 0.085 m, which the first run used and hovered at 0.072 m).
CRAWL_MAX_HEIGHT = 0.06
# Forward speed above which there is no extra reward.
CRAWL_TARGET_SPEED = 0.2

# A body part whose lowest point is closer to the floor than this (but not yet
# touching) is "about to touch down". Computed geometrically from the collision
# geoms (a floor contact margin would change MuJoCo Warp's contact physics).
TOUCHDOWN_ZONE = 0.015

# Every body with collision geoms vs. the floor.
_CONTACT_PRIMARY = ContactMatch(
  mode="body",
  pattern=(
    r"^(trunk|radius|radius_2|foot|foot_2|femur.*|tibia.*|hip_block.*"
    r"|ankle_block.*)$"
  ),
  entity="robot",
)
_FLOOR = ContactMatch(mode="body", pattern="terrain")

# Deepest contact point per body.
GROUND_CONTACT_SENSOR = ContactSensorCfg(
  name="ground_contact",
  primary=_CONTACT_PRIMARY,
  secondary=_FLOOR,
  fields=("found", "pos"),
  reduce="mindist",
  num_slots=1,
)
# Net ground reaction per body for every physics substep of the last policy step.
GROUND_FORCE_SENSOR = ContactSensorCfg(
  name="ground_force",
  primary=_CONTACT_PRIMARY,
  secondary=_FLOOR,
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  history_length=4,  # == decimation
)

##
# Observations.
##


def _heading_error(asset: Entity) -> torch.Tensor:
  """Angle between the head direction (projected on the ground) and world +x."""
  head = head_direction_w(asset)
  return torch.atan2(head[:, 1], head[:, 0])


def heading_error_sin_cos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Heading relative to the +x goal direction (IMU yaw on the real robot)."""
  err = _heading_error(env.scene[asset_cfg.name])
  return torch.stack([torch.sin(err), torch.cos(err)], dim=-1)


##
# Rewards.
##


def forward_velocity(
  env: ManagerBasedRlEnv,
  max_speed: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Trunk velocity along world +x, capped at max_speed (backward is negative)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(asset.data.root_link_lin_vel_w[:, 0], max=max_speed)


def trunk_low(
  env: ManagerBasedRlEnv, max_height: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """1 while the trunk origin stays below max_height, else 0."""
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.root_link_pos_w[:, 2] < max_height).float()


def heading_error_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  return torch.square(_heading_error(env.scene[asset_cfg.name]))


# Sensor primary index -> robot body index, resolved on first use.
_CONTACT_BODY_IDS: dict[int, list[int]] = {}


def contact_slip_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str = "ground_contact",
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Horizontal speed of each ground contact point and its in-contact mask.

  Point velocity = body linear velocity + angular velocity x (contact - origin).
  Returns (speed [B, P], in_contact [B, P]).
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  body_ids = _body_ids(asset, sensor)
  point_vel = _contact_point_vel(asset, sensor, body_ids)
  in_contact = sensor.data.found > 0
  return torch.linalg.norm(point_vel[..., :2], dim=-1), in_contact


def _contact_point_vel(asset: Entity, sensor, body_ids: list[int]) -> torch.Tensor:
  """World velocity of each body's closest-to-floor point. [B, P, 3]"""
  vel = asset.data.body_link_vel_w[:, body_ids]
  r = sensor.data.pos - asset.data.body_link_pos_w[:, body_ids]
  return vel[..., :3] + torch.cross(vel[..., 3:], r, dim=-1)


def _body_ids(asset: Entity, sensor) -> list[int]:
  body_ids = _CONTACT_BODY_IDS.get(id(sensor))
  if body_ids is None:
    body_ids = [asset.body_names.index(n) for n in sensor.primary_names]
    _CONTACT_BODY_IDS[id(sensor)] = body_ids
  return body_ids


_ROBOT_WEIGHT: dict[int, float] = {}


def robot_weight(env: ManagerBasedRlEnv) -> float:
  """Total robot weight in newtons (~8.0 N)."""
  if id(env) not in _ROBOT_WEIGHT:
    mj = env.sim.mj_model
    trunk = next(i for i in range(mj.nbody) if mj.body(i).name.endswith("trunk"))
    _ROBOT_WEIGHT[id(env)] = float(mj.body_subtreemass[trunk]) * 9.81
  return _ROBOT_WEIGHT[id(env)]


def contact_force_excess(
  env: ManagerBasedRlEnv,
  threshold: float = 1.5,
  sensor_name: str = "ground_force",
) -> torch.Tensor:
  """Sum over body parts of the peak normal force above threshold x body weight.

  Uses the per-substep force history, so impact spikes between policy steps
  count. Measured in body weights (e.g. a 3 W spike with threshold 1.5 -> 1.5).
  """
  sensor = env.scene[sensor_name]
  peak = sensor.data.force_history[..., 2].abs().amax(dim=-1)  # [B, P]
  return torch.sum(torch.relu(peak / robot_weight(env) - threshold), dim=1)


_COLLISION_POINTS: dict[int, tuple[torch.Tensor, ...]] = {}


def _collision_points(env: ManagerBasedRlEnv, asset: Entity, sensor):
  """Box corners / cylinder end centers of the contact bodies' collision geoms.

  Returns (sensor primary index [K], entity body index [K], body-frame point
  [K, 3], radius to subtract from the point height [K]).
  """
  if id(sensor) in _COLLISION_POINTS:
    return _COLLISION_POINTS[id(sensor)]
  mj = env.sim.mj_model
  short = [mj.body(i).name.split("/")[-1] for i in range(mj.nbody)]
  prim, body, pts, rad = [], [], [], []
  for g in range(mj.ngeom):
    name = short[mj.geom_bodyid[g]]
    if name not in sensor.primary_names or not (
      mj.geom_contype[g] or mj.geom_conaffinity[g]
    ):
      continue
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, mj.geom_quat[g])
    rot = rot.reshape(3, 3)
    size = mj.geom_size[g]
    if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      local = [np.array([sx, sy, sz]) * size for sx in (-1, 1) for sy in (-1, 1)
               for sz in (-1, 1)]
      r = 0.0
    else:  # capsule / cylinder: the two end centers, lowered by the radius
      local = [np.array([0.0, 0.0, h * size[1]]) for h in (-1, 1)]
      r = float(size[0])
    for p in local:
      prim.append(sensor.primary_names.index(name))
      body.append(asset.body_names.index(name))
      pts.append(mj.geom_pos[g] + rot @ p)
      rad.append(r)
  dev = env.device
  out = (
    torch.tensor(prim, device=dev),
    torch.tensor(body, device=dev),
    torch.tensor(np.array(pts), device=dev, dtype=torch.float32),
    torch.tensor(rad, device=dev, dtype=torch.float32),
  )
  _COLLISION_POINTS[id(sensor)] = out
  return out


def touchdown_speed_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str = "ground_contact",
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Squared downward speed of body parts about to touch the floor.

  A part counts when one of its collision points is within TOUCHDOWN_ZONE of
  the floor while the part has no ground contact yet; the fastest-descending
  such point per part is used, summed over parts.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  prim, body, local, radius = _collision_points(env, asset, sensor)
  pos = asset.data.body_link_pos_w[:, body]  # [B, K, 3]
  quat = asset.data.body_link_quat_w[:, body]
  vel = asset.data.body_link_vel_w[:, body]
  r = quat_apply(quat, local.expand_as(pos))
  height = pos[..., 2] + r[..., 2] - radius
  vz = vel[..., 2] + torch.cross(vel[..., 3:], r, dim=-1)[..., 2]
  touching = (sensor.data.found > 0)[:, prim]
  near = (height > 0.0) & (height < TOUCHDOWN_ZONE) & ~touching
  per_point = torch.square(torch.relu(-vz)) * near
  per_part = torch.zeros(
    per_point.shape[0], len(sensor.primary_names), device=env.device
  ).scatter_reduce(1, prim.expand_as(per_point), per_point, reduce="amax")
  return per_part.sum(dim=1)


def contact_slip_l2(
  env: ManagerBasedRlEnv, sensor_name: str = "ground_contact"
) -> torch.Tensor:
  """Sum of squared sliding speeds of the body parts touching the ground."""
  speed, in_contact = contact_slip_speed(env, sensor_name)
  return torch.sum(torch.square(speed) * in_contact, dim=1)


def lateral_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Sideways drift in the world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 1])


def yaw_rate_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Turning rate about world z."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_w[:, 2])


##
# Environment.
##


def microban_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "heading": ObservationTermCfg(
      func=heading_error_sin_cos, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0)
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "root_height": ObservationTermCfg(func=root_height),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  # Offsets around the prone pose. Targets are still clipped by the joint range
  # and torque limit defined in robot.py.
  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    ),
  }

  events = {
    "reset_prone": EventTermCfg(
      func=reset_prone,
      mode="reset",
      params={
        "yaw_range": (-0.3, 0.3),
        "tilt_range": (-0.1, 0.1),
        "joint_pos_range": (-0.15, 0.15),
      },
    ),
  }

  rewards = {
    # Main objective.
    "forward_vel": RewardTermCfg(
      func=forward_velocity,
      weight=10.0,
      params={"max_speed": CRAWL_TARGET_SPEED},
    ),
    "trunk_low": RewardTermCfg(
      func=trunk_low, weight=1.0, params={"max_height": CRAWL_MAX_HEIGHT}
    ),
    # Push off planted contacts instead of dragging limbs / the belly.
    "contact_slip": RewardTermCfg(func=contact_slip_l2, weight=-4.0),
    # Gentle contacts: per-part normal force above 1.5 x body weight, and the
    # downward speed of parts about to touch the floor. Small weights so they
    # shape the gait without blocking forward motion.
    "contact_force": RewardTermCfg(
      func=contact_force_excess, weight=-0.5, params={"threshold": 1.5}
    ),
    "touchdown_speed": RewardTermCfg(func=touchdown_speed_l2, weight=-1.0),
    # Keep going straight along +x.
    "heading": RewardTermCfg(func=heading_error_l2, weight=-1.0),
    "lateral_vel": RewardTermCfg(func=lateral_velocity_l2, weight=-5.0),
    "yaw_rate": RewardTermCfg(func=yaw_rate_l2, weight=-0.2),
    # Regularization.
    "torques": RewardTermCfg(func=joint_torques_l2, weight=-0.05),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-5.0),
    "flipped": RewardTermCfg(func=is_terminated, weight=-20.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "belly_up": TerminationTermCfg(func=belly_up, params={"threshold": 0.3}),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_microban_prone_robot_cfg()},
      sensors=(GROUND_CONTACT_SENSOR, GROUND_FORCE_SENSOR),
      num_envs=4096,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="trunk",
      distance=0.9,
      elevation=-25.0,
      azimuth=130.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=None,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,  # 50 Hz policy.
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.scene.num_envs = 16
  return cfg


##
# PPO.
##


def microban_crawl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    # Mirror augmentation + mirror loss -> left/right symmetric policy.
    algorithm=SymmetricPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="microban_crawl",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1500,
  )
