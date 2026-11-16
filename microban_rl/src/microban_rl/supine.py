"""Supine (lying on the back) initial state for Microban.

Counterpart of prone.py for tasks that start face-up (get-up from the back, ...).

Trunk frame conventions (from the MJCF): +z points from the pelvis to the head,
+x is the front of the robot (belly / toe direction), +y is the robot's left.
Lying supine means the trunk +x axis points up and, here, the head points along
world -x (feet toward world +x): a -90 deg rotation about the world y axis.
With this choice, sitting up makes the robot face world +x, the same heading as
the prone get-up keyframes (keyframes/getup_v6.lz4: crouch, stand_target).

Settled pose (CPU MuJoCo, robot.py's PD kp 0.88 / kd 0.01 / 0.64 Nm, 2 s):
  - trunk origin z = 0.031 m; the trunk leans 7 deg head-up (the back box rests
    on the floor, the heels touch); |qvel| < 0.01 at the end.
  - floor load: back/pelvis 88 %, heels 6 % + 6 % of body weight.
  - largest holding torque 0.05 Nm (hip pitch); joints sag < 0.06 rad.
Randomized spawns (yaw +-pi, tilt +-0.15 rad, joints +-0.15 rad) from
SUPINE_HEIGHT = 0.06: 0/150 start inside the floor (lowest geom 5-28 mm above),
all settle belly-up (belly z-component >= 0.98) at z = 0.029-0.033 m.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from microban_rl.robot import (
  DEFAULT_JOINT_POS,
  MICROBAN_ARTICULATION,
  get_spec,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")

# Joint pose while lying on the back: the standing pose (straight legs, arms at
# the sides), so a get-up policy can reuse the same action offset.
SUPINE_JOINT_POS: dict[str, float] = dict(DEFAULT_JOINT_POS)

# -90 deg about world y: belly (trunk +x) up, head (trunk +z) toward world -x.
SUPINE_ROOT_QUAT = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)

# The trunk settles at z = 0.031 when dropped in this pose. Spawn higher so
# that tilted (+-0.15 rad over a ~0.3 m long body) and joint-randomized poses do
# not start inside the floor; the drop is at most ~3 cm.
SUPINE_SETTLED_HEIGHT = 0.031
SUPINE_HEIGHT = 0.06

SUPINE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, SUPINE_HEIGHT),
  rot=SUPINE_ROOT_QUAT,
  joint_pos=SUPINE_JOINT_POS,
  joint_vel={".*": 0.0},
)


def get_microban_supine_robot_cfg(
  articulation: EntityArticulationInfoCfg = MICROBAN_ARTICULATION,
  spec_fn=get_spec,
) -> EntityCfg:
  """Microban with the actuators of robot.py, spawning face-up."""
  return EntityCfg(
    init_state=SUPINE_INIT_STATE,
    spec_fn=spec_fn,
    articulation=articulation,
  )


##
# Trunk axes in the world frame.
##


def _trunk_axis_w(asset: Entity, axis: int) -> torch.Tensor:
  quat = asset.data.root_link_quat_w
  vec = torch.zeros(quat.shape[0], 3, device=quat.device)
  vec[:, axis] = 1.0
  return quat_apply(quat, vec)


def belly_direction_w(asset: Entity) -> torch.Tensor:
  """World direction the belly faces (trunk +x). (0, 0, +1) when lying supine."""
  return _trunk_axis_w(asset, 0)


def head_direction_w(asset: Entity) -> torch.Tensor:
  """World direction from pelvis to head (trunk +z). (-1, 0, 0) at the default supine."""
  return _trunk_axis_w(asset, 2)


def belly_down(
  env: ManagerBasedRlEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Termination: the belly faces downward (robot rolled over onto its front).

  Note: while getting up the belly turns from up to horizontal (sitting, trunk
  vertical) and then slightly down (crouch leans forward ~20 deg), so for a
  get-up task combine this with a height/uprightness check or use a negative
  threshold.

  Args:
    threshold: Trigger when the world-z component of the belly direction is
      below -threshold. 0 means "belly at least sideways-down".
  """
  asset: Entity = env.scene[asset_cfg.name]
  return belly_direction_w(asset)[:, 2] < -threshold


##
# Reset event.
##


def reset_supine(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  tilt_range: tuple[float, float] = (-0.15, 0.15),
  xy_range: tuple[float, float] = (-0.05, 0.05),
  joint_pos_range: tuple[float, float] = (-0.15, 0.15),
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
  """Reset the robot lying supine with randomized heading, tilt and joint angles.

  Same procedure as prone.reset_prone (kept separate so the two can evolve
  independently). Orientation = world yaw * small roll/pitch tilt * the asset's
  default root quaternion (SUPINE_ROOT_QUAT when the entity uses
  get_microban_supine_robot_cfg). Joint offsets are clamped to the soft joint
  limits from robot.py.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)
  device = env.device

  root_state = asset.data.default_root_state[env_ids].clone()
  root_state[:, 0:2] += sample_uniform(*xy_range, (n, 2), device)
  root_state[:, 0:3] += env.scene.env_origins[env_ids]

  zero = torch.zeros(n, device=device)
  yaw = sample_uniform(*yaw_range, (n,), device)
  roll = sample_uniform(*tilt_range, (n,), device)
  pitch = sample_uniform(*tilt_range, (n,), device)
  q_yaw = quat_from_euler_xyz(zero, zero, yaw)
  q_tilt = quat_from_euler_xyz(roll, pitch, zero)
  root_state[:, 3:7] = quat_mul(q_yaw, quat_mul(q_tilt, root_state[:, 3:7]))
  root_state[:, 7:13] = 0.0

  asset.write_root_link_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

  joint_pos = asset.data.default_joint_pos[env_ids].clone()
  joint_pos += sample_uniform(*joint_pos_range, joint_pos.shape, device)
  limits = asset.data.soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
  joint_vel = torch.zeros_like(joint_pos)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
