"""Microban-Walk / Microban-Walk-DR: velocity-command walking for the real MERO robot.

Built on the recipe that trained microban's walk.onnx: MarcDcls/mjlab_microban
(Apache-2.0, commit d594a60, src/mjlab_microban/tasks/microban_velocity_env_cfg.py)
= mjlab's velocity task with Microban-specific rewards, DR and PPO settings
(action scale 1, init std 1, entropy 0.01, air time 0.125..0.3 s, feet
distance, pitch-10-deg upright). Our runs 1-4 with a hand-made reward (narrow
tracking std, small exploration) all settled on standing still.

Interface = walk.onnx's / mero_walk/mero/policy.py's (63 observations, 18
actions, head held at 0), so the exported ONNX is a drop-in for
`mero run --policy ... --gyro-frame body`:

    base_ang_vel (3, trunk frame), projected_gravity (3), joint_pos - default (18),
    joint_vel (18), last_action (18), command vx, vy, wz (3)
    target = default + 1.0 * action

Changes from upstream, for the MERO robot (mero_walk/README.md):
* 30 Hz control (115200 bps bus): physics dt 1/180 s, decimation 6.
* Motors on a 5 V regulator, kp_walk 190 register ~ 0.275 Nm/rad effective
  (upstream: BAM model, kp_fw 125 at 7-8 V); ideal PD with the MJCF back-EMF
  damping, torque limit 0.64 Nm, command delay 2..5 physics steps (11..28 ms).
* Our MJCF (microban submodule, all collision geoms as in robot.xml).

Microban-Walk-DR adds (floor_dr.py, backlash.py):
* Floor friction 0.4..1.2 and contact softness per env per reset.
* Gear backlash 0..3 deg per joint (the actor sees the encoder angle),
  calibration error +-2 deg, kp x0.75..1.3 and torque limit x0.75..1.0.
"""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import root_height_below_minimum
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from microban_rl import symmetry
from microban_rl.backlash import BacklashPdActuatorCfg, joint_pos_rel_sensed
from microban_rl.floor_dr import floor_events, get_spec_floor_dr
from microban_rl.robot import INIT_STATE, XL330_DAMPING, XL330_EFFORT_LIMIT, get_spec
from microban_rl.symmetry import SymmetricPpoAlgorithmCfg
from microban_rl.walk_onnx import KP_RL_STIFFNESS, WALK_JOINTS

CONTROL_HZ = 30
DECIMATION = 6
PHYSICS_DT = 1.0 / (CONTROL_HZ * DECIMATION)
ACTION_SCALE = 1.0
KP_WALK_REGISTER_5V = 190  # mero_walk config.5v.example.json control.kp_walk
CMD_DELAY_STEPS = (2, 5)  # physics steps (11..28 ms)

FOOT_SITES = ("right_foot", "left_foot")  # same order as the bodies foot (right), foot_2 (left)
WALKING_THRESHOLD = 0.01

_WALK = SceneEntityCfg("robot", joint_names=WALK_JOINTS, preserve_order=True)

# Mirror rules for the observation terms used here.
symmetry._TERM_SIGNS.setdefault("command", (1.0, -1.0, -1.0))
for _t in ("foot_height", "foot_air_time", "foot_contact"):
  symmetry._TERM_MAPS.setdefault(_t, ((1, 0), (1.0, 1.0)))


def base_ang_vel_trunk(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Trunk-frame angular velocity (= mero_walk gyro_frame body)."""
  return env.scene["robot"].data.root_link_ang_vel_b


##
# Reward terms from mjlab_microban (Apache-2.0, Marc Duclusaud).
##


class upright_pitch:
  """exp(-|g_xy - (sin(pitch), 0)|^2 / std^2): trunk upright at a forward pitch (mjlab_microban `upright`)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    pass

  def __call__(self, env: ManagerBasedRlEnv, std: float, pitch: float = 0.0) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    g = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
    g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    err = (g[:, 0] - math.sin(pitch)).square() + g[:, 1].square()
    return torch.exp(-err / std**2)

  def reset(self, env_ids=None) -> None:
    pass


def feet_distance_penalty(env: ManagerBasedRlEnv, min_dist: float) -> torch.Tensor:
  """clamp(min_dist - horizontal foot-site distance, 0) (mjlab_microban)."""
  asset: Entity = env.scene["robot"]
  ids, _ = asset.find_sites(FOOT_SITES, preserve_order=True)
  p = asset.data.site_pos_w[:, ids, :2]
  return torch.clamp(min_dist - torch.norm(p[:, 0] - p[:, 1], dim=-1), min=0.0)


##
# Config.
##


def _actuator(dr_on: bool) -> BacklashPdActuatorCfg:
  return BacklashPdActuatorCfg(
    target_names_expr=("head", ".*_shoulder_.*", ".*_elbow", ".*_hip_.*", ".*_knee", ".*_ankle_.*"),
    stiffness=KP_RL_STIFFNESS,
    damping=XL330_DAMPING,
    effort_limit=XL330_EFFORT_LIMIT,
    backlash_range=(0.0, math.radians(3.0)) if dr_on else (0.0, 0.0),
    offset_range=(-math.radians(2.0), math.radians(2.0)) if dr_on else (0.0, 0.0),
    kp_scale_range=(0.75, 1.3) if dr_on else (0.9, 1.1),
    effort_scale_range=(0.75, 1.0) if dr_on else (0.9, 1.0),
    delay_min_lag=CMD_DELAY_STEPS[0],
    delay_max_lag=CMD_DELAY_STEPS[1],
  )


def microban_walk_env_cfg(play: bool = False, dr_on: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim = SimulationCfg(
    nconmax=None, njmax=None,
    mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20, ccd_iterations=100),
  )
  cfg.decimation = DECIMATION
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="trunk",
    distance=0.8, elevation=-15.0, azimuth=90.0,
  )

  # ---- scene / sensors ----
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(foot|foot_2)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  robot = EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec_floor_dr if dr_on else get_spec,
    articulation=EntityArticulationInfoCfg(actuators=(_actuator(dr_on),), soft_joint_pos_limit_factor=0.9),
  )
  cfg.scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot},
    sensors=(feet_ground, foot_height_scan, self_collision),
    num_envs=4096,
    env_spacing=1.0,
  )

  # ---- actions: 18 joints (no head), scale 1 ----
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot", actuator_names=WALK_JOINTS, scale=ACTION_SCALE, use_default_offset=True,
    preserve_order=True,
  )

  # ---- observations (walk.onnx layout, mjlab_microban noise / delays) ----
  actor = cfg.observations["actor"].terms
  actor["base_ang_vel"] = ObservationTermCfg(
    func=base_ang_vel_trunk, noise=Unoise(n_min=-0.03, n_max=0.03),
    delay_min_lag=0, delay_max_lag=2, delay_update_period=64,
  )
  actor["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01),
    delay_min_lag=0, delay_max_lag=2, delay_update_period=64,
  )
  actor["joint_pos"] = ObservationTermCfg(
    func=joint_pos_rel_sensed, params={"asset_cfg": _WALK}, noise=Unoise(n_min=-0.001, n_max=0.001),
  )
  actor["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel, params={"asset_cfg": _WALK}, noise=Unoise(n_min=-0.25, n_max=0.25),
    delay_min_lag=0, delay_max_lag=1,
  )
  order = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command")
  cfg.observations["actor"].terms = {k: actor[k] for k in order}
  cfg.observations["critic"].terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel_trunk),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": _WALK}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": _WALK}),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    "foot_height": ObservationTermCfg(func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}),
    "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
  }

  # ---- rewards (mjlab_microban values) ----
  r = cfg.rewards
  r["track_linear_velocity"].params["std"] = math.sqrt(0.1)
  r["track_linear_velocity"].weight = 2.0
  r["track_angular_velocity"].params["std"] = math.sqrt(0.5)
  r["track_angular_velocity"].weight = 2.0
  std_standing = {
    r".*shoulder_pitch.*": 0.1, r".*shoulder_roll.*": 0.1, r".*elbow.*": 0.1,
    r".*hip_roll.*": 0.1, r".*hip_pitch.*": 0.15, r".*hip_yaw.*": 0.1,
    r".*knee.*": 0.15, r".*ankle_pitch.*": 0.1, r".*ankle_roll.*": 0.1,
  }
  std_walking = {
    r".*shoulder_pitch.*": 0.4, r".*shoulder_roll.*": 0.2, r".*elbow.*": 0.2,
    r".*hip_roll.*": 0.2, r".*hip_pitch.*": 0.4, r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.4, r".*ankle_pitch.*": 0.3, r".*ankle_roll.*": 0.2,
  }
  p = r["pose"].params
  p["asset_cfg"] = _WALK
  p["std_standing"], p["std_walking"], p["std_running"] = std_standing, std_walking, std_walking
  p["walking_threshold"] = WALKING_THRESHOLD
  r["pose"].weight = 1.0
  r["upright"] = RewardTermCfg(func=upright_pitch, weight=1.0,
                               params={"std": math.sqrt(0.1), "pitch": math.radians(10.0)})
  r["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
  r["body_ang_vel"].weight = -0.05
  r.pop("angular_momentum")
  for name in ("foot_clearance", "foot_slip"):
    r[name].params["asset_cfg"].site_names = FOOT_SITES
  r["foot_clearance"].params["command_threshold"] = WALKING_THRESHOLD
  r["foot_clearance"].params["target_height"] = 0.02
  r["foot_swing_height"].params["command_threshold"] = WALKING_THRESHOLD
  r["foot_swing_height"].params["target_height"] = 0.02
  r["air_time"].params["command_threshold"] = WALKING_THRESHOLD
  r["air_time"].params["threshold_min"] = 0.125
  r["air_time"].params["threshold_max"] = 0.300
  r["air_time"].weight = 3.0
  r.pop("soft_landing")
  r["foot_slip"].params["command_threshold"] = WALKING_THRESHOLD
  r["foot_slip"].weight = -1.0
  r["action_rate_l2"].weight = -0.1
  r["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
                                       params={"sensor_name": self_collision.name})
  r["feet_distance"] = RewardTermCfg(func=feet_distance_penalty, weight=-1000.0, params={"min_dist": 0.08})

  # ---- commands (MERO teleop range: vx -0.2..0.3, turn +-0.6) ----
  cmd = cfg.commands["twist"]
  cmd.rel_standing_envs = 0.1
  cmd.rel_heading_envs = 0.0
  cmd.rel_forward_envs = 0.0
  cmd.heading_command = False
  cmd.debug_vis = False
  cmd.ranges.lin_vel_x = (-0.3, 0.4)
  cmd.ranges.lin_vel_y = (-0.2, 0.2)
  cmd.ranges.ang_vel_z = (-0.75, 0.75)
  cmd.ranges.heading = None

  # ---- events ----
  ev = cfg.events
  ev["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)
  ev["push_robot"].params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
  ev.pop("encoder_bias")  # calibration error is the actuator's offset_range (DR variant)
  ev["base_com"].params["ranges"] = {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)}
  ev["base_com"].params["asset_cfg"].body_names = ("trunk",)
  ev["dof_armature_randomization"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale", "ranges": (0.9, 1.1)},
  )
  ev["dof_friction_randomization"] = EventTermCfg(
    mode="startup", func=dr.joint_friction,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale", "ranges": (0.9, 1.1)},
  )
  if dr_on:
    ev.pop("foot_friction")
    ev.update(floor_events())
  else:
    ev["foot_friction"].params["asset_cfg"].geom_names = (r".*_foot_collision_.*",)

  # ---- terminations / curriculum ----
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=root_height_below_minimum, params={"minimum_height": 0.10}),
  }
  cfg.curriculum = {}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.events.pop("push_robot")
    cfg.scene.num_envs = 16
  return cfg


def microban_walk_dr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microban_walk_env_cfg(play=play, dr_on=True)


def microban_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=SymmetricPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="microban_walk",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3000,
  )


def microban_walk_dr_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fine-tuning from walk.onnx (scripts/init_from_walk_onnx.py): small entropy bonus so the
  0.15 initial action std is not inflated before the fresh critic has caught up."""
  cfg = microban_walk_ppo_runner_cfg()
  cfg.algorithm.entropy_coef = 0.002
  cfg.max_iterations = 2000
  cfg.run_name = "dr"
  return cfg


##
# Microban-Walk-DR2: low-current fine-tune (real robot runs, 2026-11-20).
##

# XL330 torque constant (microban BAM constants) for the motor-current proxy.
XL330_KT = 0.366  # Nm/A


def no_stepping(env: ManagerBasedRlEnv, command_threshold: float = 0.02) -> torch.Tensor:
  """Feet in the air while the command is ~0 (mjlab_microban `no_stepping_penalty`)."""
  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  still = (torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()) < command_threshold
  found = env.scene["feet_ground_contact"].data.found
  return (found <= 0).float().sum(1) * still.float()


def motor_current(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over all motors of |torque| / kt (A): proxy of the regulator load."""
  return env.scene["robot"].data.actuator_force.abs().sum(1) / XL330_KT


def microban_walk_dr2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Microban-Walk-DR plus, from the real MERO runs:

  * The 5 V regulator browns out (robot reboots while walking or even standing) and loaded
    joints sag: walking needs ~4 A mean / 6-7 A peaks of motor current in sim, and DR
    model_1000 steps in place at zero command (3.5 A vs 0.9 A for walk.onnx). So:
    no_stepping (-1, feet off the ground at zero command), motor_current (-0.1 per A),
    20 % zero commands.
  * XL330 at 5 V: datasheet stall torque 0.52 Nm (sim limit 0.64 Nm), supply sagging to
    ~4.6 V under load: torque limit x0.6..0.95.
  """
  cfg = microban_walk_env_cfg(play=play, dr_on=True)
  act = cfg.scene.entities["robot"].articulation.actuators[0]
  assert isinstance(act, BacklashPdActuatorCfg)
  act.effort_scale_range = (0.6, 0.95)
  cfg.rewards["no_stepping"] = RewardTermCfg(func=no_stepping, weight=-1.0)
  cfg.rewards["motor_current"] = RewardTermCfg(func=motor_current, weight=-0.1)
  if not play:
    cfg.commands["twist"].rel_standing_envs = 0.2
  return cfg


def microban_walk_dr2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = microban_walk_dr_ppo_runner_cfg()
  cfg.max_iterations = 1000
  cfg.run_name = "dr2"
  return cfg
