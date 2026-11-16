"""Microban "stand without falling" task.

The policy outputs joint position offsets around the default pose. It is
rewarded for staying alive, upright and at standing height, and is pushed
around periodically so that it actually has to balance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  is_alive,
  is_terminated,
  joint_pos_limits,
  joint_pos_rel,
  joint_torques_l2,
  joint_vel_l2,
  joint_vel_rel,
  last_action,
  projected_gravity,
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
  root_height_below_minimum,
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
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from microban_rl.robot import INIT_HEIGHT, get_microban_robot_cfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")

# Trunk height while standing in the default pose (feet flat on the ground).
STAND_HEIGHT = INIT_HEIGHT - 0.002

##
# Custom rewards.
##


def upright_exp(
  env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """1 when the trunk is perfectly upright, decaying with tilt."""
  asset: Entity = env.scene[asset_cfg.name]
  tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-tilt_sq / std**2)


def base_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """1 when the trunk is at the standing height, decaying with the error."""
  asset: Entity = env.scene[asset_cfg.name]
  err = asset.data.root_link_pos_w[:, 2] - target_height
  return torch.exp(-torch.square(err) / std**2)


def default_pose_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Squared deviation of the joints from the default standing pose."""
  asset: Entity = env.scene[asset_cfg.name]
  err = asset.data.joint_pos - asset.data.default_joint_pos
  return torch.sum(torch.square(err), dim=1)


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


##
# Environment.
##


def microban_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0)
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  # The critic additionally sees privileged, noise-free state.
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "root_height": ObservationTermCfg(func=root_height),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    ),
    "push_robot": EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0),
      params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    ),
  }

  rewards = {
    # "Stand without falling": the main positive signals.
    "alive": RewardTermCfg(func=is_alive, weight=1.0),
    "upright": RewardTermCfg(func=upright_exp, weight=1.0, params={"std": 0.3}),
    "base_height": RewardTermCfg(
      func=base_height_exp,
      weight=1.0,
      params={"target_height": STAND_HEIGHT, "std": 0.02},
    ),
    # Falling ends the episode with a large penalty.
    "fall": RewardTermCfg(func=is_terminated, weight=-50.0),
    # Regularization: stay close to the nominal pose, move smoothly.
    "default_pose": RewardTermCfg(func=default_pose_l2, weight=-0.5),
    "joint_vel": RewardTermCfg(func=joint_vel_l2, weight=-1e-3),
    "torques": RewardTermCfg(func=joint_torques_l2, weight=-0.1),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
    "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-5.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=bad_orientation, params={"limit_angle": 0.8}
    ),
    "too_low": TerminationTermCfg(
      func=root_height_below_minimum, params={"minimum_height": 0.10}
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_microban_robot_cfg()},
      num_envs=4096,
      env_spacing=0.6,
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
      distance=0.8,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # Contact/constraint buffers sized by MJLab's heuristic. The foot boxes
      # alone produce ~32 contacts per world, and more during falls.
      nconmax=None,
      njmax=None,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,  # 50 Hz policy.
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.scene.num_envs = 16
  return cfg


##
# PPO.
##


def microban_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="microban_stand",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1500,
  )
