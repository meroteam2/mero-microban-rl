"""Microban "get up from prone" task (Microban-GetUp), imitating keyframes made in robot_keyframe_kit.

Reference: data/getup_reference.npz, built by scripts/make_getup_reference.py
from the training copy of the user's keyframes (getup_v6 with the
pike_push -> fold interval stretched to 0.8 s): prone -> arm_prep -> chest_up
-> plank -> plank_1 -> pike -> pike_1 -> pike_push -> fold -> squat_deep ->
crouch -> stand_target.

- Action: residual around the reference. Joint target = q_ref(t + dt) +
  ACTION_SCALE * action, through the 10.3 rad/s target speed limit
  (SpeedLimitedJointPositionAction), robot.py's actuators (kp 0.88, 0.64 Nm).
  Open loop the reference alone fails at the hand push (pike_push -> fold): the
  PD cannot make the ~0.4 Nm knee / ankle torque, so the policy has to command
  past the reference.
- Observations: IMU-like terms, joint pos / vel, last action, the phase
  (reference time / end time) and the next reference joint targets.
- Starts (reference-state initialization): RSI_PRONE prone (prone.py
  randomization), RSI_REPLAY states of the open-loop replay up to pike_push
  (weighted toward pike -> pike_push), RSI_KEYFRAME the keyframe poses
  pike_1 ... crouch at rest. Random heading for all.
- Rewards: pose / trunk imitation (tolerance x3 over pike_1 -> squat_deep, so
  the policy chooses the push itself), capture-point push shaping (potential
  based), weight-transfer bonus, height / upright, stand-target bonus,
  face-plant / knee-contact penalties, regularization ramped 0.1 -> 1 over
  REG_RAMP_ITERATIONS.
- Termination: flipped onto the back (belly_up) or the reference time passing
  the end + HOLD_S. No fall termination.
- Mirror symmetry (symmetry.py augmentation + mirror loss).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

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
)
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from microban_rl import symmetry
from microban_rl.crawl_arms_env_cfg import (
  STEPS_PER_ITERATION,
  SpeedLimitedJointPositionAction,
  SpeedLimitedJointPositionActionCfg,
  steps_since_start,
)
from microban_rl.crawl_env_cfg import contact_slip_l2
from microban_rl.prone import belly_up, get_microban_prone_robot_cfg, reset_prone
from microban_rl.robot import DEFAULT_JOINT_POS
from microban_rl.stand_env_cfg import root_height
from microban_rl.symmetry import SymmetricPpoAlgorithmCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAIN_KEYFRAMES = str(DATA_DIR / "getup_v6_train.lz4")
REFERENCE_FILE = str(DATA_DIR / "getup_reference.npz")

ACTION_SCALE = 0.5  # rad of residual per unit action
HOLD_S = 3.0  # episode continues this long after the last keyframe
PUSH_WINDOW = ("pike_1", "squat_deep")  # wide imitation tolerance here
WIDE = 3.0  # tolerance factor in the push window
POSE_SIGMA = 0.3  # rad, RMS joint error scale
TILT_SIGMA = 0.35  # rad
HEIGHT_SIGMA = 0.03  # m
REG_START, REG_RAMP_ITERATIONS = 0.1, 500
RSI_PRONE, RSI_REPLAY, RSI_KEYFRAME = 0.4, 0.35, 0.25
RSI_KEYFRAMES = ("pike_1", "pike_push", "fold", "squat_deep", "crouch")
PRONE_T0 = -0.5  # prone starts settle for 0.5 s on the prone keyframe

# Stand target = the walk.onnx start pose approved in stage 1.
STAND_PITCH = (0.0, 0.2)
STAND_ROLL = 0.1
STAND_JOINT_RMS = 0.12
STAND_QVEL = 1.0  # rad/s, "nearly still"
STAND_LIN_VEL = 0.1  # m/s
STAND_HOLD_S = 1.0
SOLE_FLAT_COS = math.cos(0.15)

# Ground contacts per body. Bodies without collision geoms never report one.
GETUP_CONTACT_SENSOR = ContactSensorCfg(
  name="getup_contact",
  primary=ContactMatch(
    mode="body",
    pattern=(r"^(trunk|head|shoulder|shoulder_2|humerus.*|radius|radius_2|hip|hip_2|hip_block.*"
             r"|femur.*|tibia.*|ankle_block.*|foot|foot_2)$"),
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "pos"),
  reduce="mindist",
  num_slots=1,
)
BAD_CONTACT = ("trunk", "head", "shoulder", "shoulder_2", "hip", "hip_2", "hip_block__configuration_left",
               "hip_block__configuration_right", "humerus__configuration_left", "humerus__configuration_right")
KNEE_CONTACT = ("femur__configuration_left", "femur__configuration_right", "tibia__configuration_left",
                "tibia__configuration_right", "ankle_block__configuration_left", "ankle_block__configuration_right")

# The phase and reference observations mirror like a scalar and joint targets.
symmetry._TERM_SIGNS.setdefault("getup_phase", (1.0,))
symmetry._JOINT_TERMS.add("ref_joint_pos")


##
# Reference.
##


class Reference:
  """getup_reference.npz on the env's device, columns in the robot's joint order."""

  def __init__(self, env: ManagerBasedRlEnv):
    d = np.load(REFERENCE_FILE)
    robot = env.scene["robot"]
    names = [n.split("/")[-1] for n in robot.joint_names]
    cols = [list(d["joint_names"]).index(n) for n in names]
    dev = env.device
    f = lambda x: torch.tensor(np.asarray(x), device=dev, dtype=torch.float32)  # noqa: E731
    self.dt = float(d["t"][1] - d["t"][0])
    self.q = f(d["q_ref"][:, cols])
    self.tilt, self.z = f(d["tilt_ref"]), f(d["z_ref"])
    self.t_end = float(d["t_end"])
    self.physical_until = float(d["physical_until"])
    self.kf_names = [str(x) for x in d["kf_names"]]
    self.kf_t = dict(zip(self.kf_names, (float(x) for x in d["kf_t"])))
    self.kf_root = f(d["kf_root"])
    self.kf_joint = f(d["kf_joint"][:, cols])
    self.replay = {k[len("replay_"):]: f(d[k]) for k in d.files if k.startswith("replay_")}
    self.replay["joint_pos"] = self.replay["joint_pos"][:, cols]
    self.replay["joint_vel"] = self.replay["joint_vel"][:, cols]
    self.default = f([DEFAULT_JOINT_POS[n] for n in names])
    self.window = (self.kf_t[PUSH_WINDOW[0]], self.kf_t[PUSH_WINDOW[1]])

  def index(self, t: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(t / self.dt).long(), 0, self.q.shape[0] - 1)


def reference(env: ManagerBasedRlEnv) -> Reference:
  ref = env.__dict__.get("_getup_ref")
  if ref is None:
    ref = Reference(env)
    env._getup_ref = ref  # type: ignore[attr-defined]
    env._getup_t0 = torch.full((env.num_envs,), PRONE_T0, device=env.device)  # type: ignore[attr-defined]
    # Own step counter: the train script randomizes episode_length_buf at start.
    env._getup_steps = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
  return ref


def elapsed_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Policy steps since this env's reset (advanced when the step's action is processed)."""
  reference(env)
  return env._getup_steps


def ref_time(env: ManagerBasedRlEnv, ahead: float = 0.0) -> torch.Tensor:
  """Reference time of each env [B]: start offset + elapsed time (+ ahead)."""
  return env._getup_t0 + elapsed_steps(env) * env.step_dt + ahead


def in_window(env: ManagerBasedRlEnv) -> torch.Tensor:
  lo, hi = reference(env).window
  t = ref_time(env)
  return (t >= lo) & (t <= hi)


##
# Action: residual around the reference.
##


class ReferenceResidualActionCfg(SpeedLimitedJointPositionActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> ReferenceResidualAction:
    return ReferenceResidualAction(self, env)


class ReferenceResidualAction(SpeedLimitedJointPositionAction):
  """target = q_ref(t + dt) + scale * action, then the target speed limit."""

  def process_actions(self, actions: torch.Tensor) -> None:
    ref = reference(self._env)
    if not hasattr(self, "_ref_cols"):
      names = [n.split("/")[-1] for n in self._entity.joint_names]
      self._ref_cols = torch.tensor(
        [names.index(n.split("/")[-1]) for n in self.target_names], device=self.device)
    self._env._getup_steps += 1  # this step's targets are for its end
    q = ref.q[ref.index(ref_time(self._env))]
    self._offset = q[:, self._ref_cols]
    super().process_actions(actions)


##
# Observations.
##


def getup_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  ref = reference(env)
  return torch.clamp(ref_time(env) / ref.t_end, 0.0, 1.0).unsqueeze(-1)


def ref_joint_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Next reference joint targets relative to the default pose (robot joint order)."""
  ref = reference(env)
  return ref.q[ref.index(ref_time(env, env.step_dt))] - ref.default


##
# Reset: reference-state initialization.
##


def _yaw_rotate(pos, quat, lin, ang, yaw):
  zero = torch.zeros_like(yaw)
  qz = quat_from_euler_xyz(zero, zero, yaw)
  return quat_apply(qz, pos), quat_mul(qz, quat), quat_apply(qz, lin), quat_apply(qz, ang)


def reset_getup(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  p_prone: float = RSI_PRONE,
  p_replay: float = RSI_REPLAY,
  joint_noise: float = 0.05,
  prone_params: dict | None = None,
  keyframe_weights: dict[str, float] | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
  """keyframe_weights: relative start weight per RSI keyframe (default: uniform over RSI_KEYFRAMES)."""
  env_ids = resolve_env_ids(env, env_ids)
  ref = reference(env)
  asset: Entity = env.scene[asset_cfg.name]
  dev, n = env.device, len(env_ids)
  u = torch.rand(n, device=dev)
  prone = u < p_prone
  replay = (u >= p_prone) & (u < p_prone + p_replay)
  keyf = u >= p_prone + p_replay

  if prone.any():
    ids = env_ids[prone]
    reset_prone(env, ids, **(prone_params or {}))
    env._getup_t0[ids] = PRONE_T0
  env._getup_steps[env_ids] = 0.0

  others = ~prone
  if others.any():
    ids = env_ids[others]
    m = int(others.sum())
    is_rep = replay[others]
    # Replay states: half uniform over [0, physical_until], half over [pike, physical_until].
    t_hi = ref.physical_until
    t_lo = torch.where(torch.rand(m, device=dev) < 0.5, torch.zeros(m, device=dev),
                       torch.full((m,), ref.kf_t["pike"], device=dev))
    t_rep = t_lo + (t_hi - t_lo) * torch.rand(m, device=dev)
    i_rep = ref.index(t_rep).clamp(max=ref.replay["root_pos"].shape[0] - 1)
    kw = keyframe_weights or {k: 1.0 for k in RSI_KEYFRAMES}
    k_ids = torch.tensor([ref.kf_names.index(k) for k in kw], device=dev)
    k_w = torch.tensor(list(kw.values()), device=dev, dtype=torch.float32)
    k_pick = k_ids[torch.multinomial(k_w, m, replacement=True)]
    rp = ref.replay
    pos = torch.where(is_rep[:, None], rp["root_pos"][i_rep], ref.kf_root[k_pick, :3] + torch.tensor([0, 0, 0.002], device=dev))
    quat = torch.where(is_rep[:, None], rp["root_quat"][i_rep], ref.kf_root[k_pick, 3:7])
    lin = torch.where(is_rep[:, None], rp["root_lin_vel"][i_rep], torch.zeros(m, 3, device=dev))
    ang = torch.where(is_rep[:, None], rp["root_ang_vel"][i_rep], torch.zeros(m, 3, device=dev))
    qj = torch.where(is_rep[:, None], rp["joint_pos"][i_rep], ref.kf_joint[k_pick])
    vj = torch.where(is_rep[:, None], rp["joint_vel"][i_rep], torch.zeros_like(qj))
    kf_t = torch.tensor([ref.kf_t[ref.kf_names[i]] for i in range(len(ref.kf_names))], device=dev)
    t0 = torch.where(is_rep, ref.dt * i_rep.float(), kf_t[k_pick])
    yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
    pos, quat, lin, ang = _yaw_rotate(pos, quat, lin, ang, yaw)
    pos = pos + env.scene.env_origins[ids]
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], -1), env_ids=ids)
    asset.write_root_link_velocity_to_sim(torch.cat([lin, ang], -1), env_ids=ids)
    qj = qj + (torch.rand_like(qj) * 2 - 1) * joint_noise
    lim = asset.data.soft_joint_pos_limits[ids]
    qj = qj.clamp(lim[..., 0], lim[..., 1])
    asset.write_joint_state_to_sim(qj, vj, env_ids=ids)
    env._getup_t0[ids] = t0


##
# Shared per-step quantities (support, capture point, contacts).
##


class _Geometry:
  def __init__(self, env: ManagerBasedRlEnv):
    import mujoco

    robot = env.scene["robot"]
    mj = env.sim.mj_model
    bnames = list(robot.body_names)
    full = {mj.body(i).name.split("/")[-1]: i for i in range(mj.nbody)}
    self.mass = torch.tensor([mj.body_mass[full[b]] for b in bnames], device=env.device)
    self.feet = [bnames.index(b) for b in ("foot_2", "foot")]  # left, right
    pts, normals = [], []
    for side in ("left", "right"):
      g = [mj.geom(f"robot/{side}_foot_collision_{i}").id if _has(mj, f"robot/{side}_foot_collision_{i}")
           else mj.geom(f"{side}_foot_collision_{i}").id for i in (1, 3, 4, 5, 6)]
      pts.append(np.array([mj.geom_pos[i] for i in g[1:]]))  # heel 3,4  toe 5,6 (body frame)
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, mj.geom_quat[g[0]])
      normals.append(rot.reshape(3, 3)[:, 2])
    self.corners = torch.tensor(np.array(pts), device=env.device, dtype=torch.float32)  # [2, 4, 3]
    self.normal = torch.tensor(np.array(normals), device=env.device, dtype=torch.float32)  # [2, 3]
    toe_dir = self.corners[:, 2:].mean(1) - self.corners[:, :2].mean(1)
    self.toe_dir = toe_dir / toe_dir.norm(dim=-1, keepdim=True)  # [2, 3] body frame
    sensor = env.scene[GETUP_CONTACT_SENSOR.name]
    pn = list(sensor.primary_names)
    pick = lambda names: torch.tensor([pn.index(b) for b in names if b in pn], device=env.device)  # noqa: E731
    self.c_feet, self.c_hands = pick(("foot_2", "foot")), pick(("radius_2", "radius"))
    self.c_bad, self.c_knee = pick(BAD_CONTACT), pick(KNEE_CONTACT)


def _has(mj, name: str) -> bool:
  try:
    mj.geom(name)
    return True
  except KeyError:
    return False


def getup_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Support / balance quantities of the current step (cached per physics step)."""
  key = env._sim_step_counter
  cache = env.__dict__.get("_getup_state")
  if cache is not None and cache[0] == key:
    return cache[1]
  geo = env.__dict__.get("_getup_geo")
  if geo is None:
    geo = env._getup_geo = _Geometry(env)  # type: ignore[attr-defined]
  robot: Entity = env.scene["robot"]
  d = robot.data
  m = geo.mass
  com = (d.body_com_pos_w * m[:, None]).sum(1) / m.sum()
  vcom = (d.body_com_lin_vel_w * m[:, None]).sum(1) / m.sum()
  fq = d.body_link_quat_w[:, geo.feet]  # [B, 2, 4]
  fp = d.body_link_pos_w[:, geo.feet]
  B = fq.shape[0]
  corners = fp[:, :, None] + quat_apply(fq[:, :, None].expand(B, 2, 4, 4).reshape(-1, 4),
                                       geo.corners[None].expand(B, 2, 4, 3).reshape(-1, 3)).reshape(B, 2, 4, 3)
  toe = quat_apply(fq.reshape(-1, 4), geo.toe_dir[None].expand(B, 2, 3).reshape(-1, 3)).reshape(B, 2, 3)
  normal = quat_apply(fq.reshape(-1, 4), geo.normal[None].expand(B, 2, 3).reshape(-1, 3)).reshape(B, 2, 3)
  f = toe[..., :2].mean(1)
  f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # forward (toe) direction on the ground
  proj = (corners[..., :2] * f[:, None, None]).sum(-1)  # [B, 2, 4]
  heel = proj[..., :2].amin((1, 2))
  toe_x = proj[..., 2:].amax((1, 2))
  com_x = (com[:, :2] * f).sum(-1)
  omega = torch.sqrt(9.81 / com[:, 2].clamp(min=0.03))
  cp = com_x + (vcom[:, :2] * f).sum(-1) / omega
  found = env.scene[GETUP_CONTACT_SENSOR.name].data.found > 0
  q = robot.data.projected_gravity_b
  ez = torch.tensor([0, 0, 1.0], device=env.device).expand(B, 3)
  head = quat_apply(d.root_link_quat_w, ez)
  st = {
    "com": com, "vcom": vcom, "heel": heel, "toe": toe_x, "com_x": com_x, "cp": cp,
    "feet_down": found[:, geo.c_feet].all(1),
    "soles_flat": (normal[..., 2] > SOLE_FLAT_COS).all(1),
    "hands_down": found[:, geo.c_hands].any(1),
    "bad_contact": found[:, geo.c_bad].any(1) if len(geo.c_bad) else torch.zeros(B, dtype=torch.bool, device=env.device),
    "knee_contact": found[:, geo.c_knee].any(1) if len(geo.c_knee) else torch.zeros(B, dtype=torch.bool, device=env.device),
    "tilt": torch.acos(head[:, 2].clamp(-1, 1)),
    "pitch": -torch.atan2(-q[:, 0], -q[:, 2]),
    "roll": torch.atan2(q[:, 1], -q[:, 2]),
    "z": d.root_link_pos_w[:, 2],
  }
  env._getup_state = (key, st)  # type: ignore[attr-defined]
  return st


def stand_ok(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The approved stand-target / walk.onnx start conditions (stage 1), per env."""
  st = getup_state(env)
  robot: Entity = env.scene["robot"]
  ref = reference(env)
  d = robot.data
  names = [n.split("/")[-1] for n in robot.joint_names]
  walk = torch.tensor([i for i, n in enumerate(names) if n != "head"], device=env.device)
  rms = (d.joint_pos[:, walk] - ref.default[walk]).square().mean(1).sqrt()
  return ((st["pitch"] >= STAND_PITCH[0]) & (st["pitch"] <= STAND_PITCH[1]) & (st["roll"].abs() <= STAND_ROLL)
          & st["feet_down"] & st["soles_flat"] & ~st["hands_down"] & ~st["bad_contact"] & ~st["knee_contact"]
          & (rms < STAND_JOINT_RMS) & (d.joint_vel.abs().amax(1) < STAND_QVEL)
          & (d.root_link_lin_vel_w.norm(dim=-1) < STAND_LIN_VEL))


##
# Rewards.
##


def _wide(env, base: float) -> torch.Tensor:
  return torch.where(in_window(env), base * WIDE, base)


def pose_imitation(env: ManagerBasedRlEnv) -> torch.Tensor:
  ref = reference(env)
  q = env.scene["robot"].data.joint_pos
  err = (q - ref.q[ref.index(ref_time(env))]).square().mean(1)
  return torch.exp(-err / _wide(env, POSE_SIGMA) ** 2)


def trunk_imitation(env: ManagerBasedRlEnv) -> torch.Tensor:
  ref = reference(env)
  st = getup_state(env)
  i = ref.index(ref_time(env))
  a = torch.exp(-(st["tilt"] - ref.tilt[i]).square() / _wide(env, TILT_SIGMA) ** 2)
  b = torch.exp(-(st["z"] - ref.z[i]).square() / _wide(env, HEIGHT_SIGMA) ** 2)
  return 0.5 * (a + b)


def _support_potential(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-(distance of the capture point outside the soles), when both feet are down after pike_1."""
  st = getup_state(env)
  out = torch.relu(st["cp"] - st["toe"]) + torch.relu(st["heel"] - st["cp"])
  active = st["feet_down"] & (ref_time(env) >= reference(env).kf_t[PUSH_WINDOW[0]])
  return torch.where(active, -out.clamp(max=0.15), torch.zeros_like(out))


def push_shaping(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Potential-based shaping (Phi_t - Phi_{t-1}) / dt with Phi = _support_potential."""
  phi = _support_potential(env)
  prev = env.__dict__.get("_getup_phi")
  if prev is None:
    prev = phi.clone()
  fresh = elapsed_steps(env) <= 1
  r = torch.where(fresh, torch.zeros_like(phi), (phi - prev) / env.step_dt)
  env._getup_phi = phi.clone()  # type: ignore[attr-defined]
  return r


def weight_transfer(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while standing on the flat soles only, capture point inside them (after pike_push)."""
  st = getup_state(env)
  inside = (st["cp"] <= st["toe"]) & (st["cp"] >= st["heel"])
  late = ref_time(env) >= reference(env).kf_t["pike_push"]
  return (st["feet_down"] & st["soles_flat"] & ~st["hands_down"] & ~st["bad_contact"] & inside & late).float()


def upright_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """cos(tilt) x height / stand height, from squat_deep on, without trunk contact."""
  st = getup_state(env)
  late = ref_time(env) >= reference(env).kf_t["squat_deep"]
  v = torch.cos(st["tilt"]).clamp(min=0.0) * (st["z"] / 0.17).clamp(0.0, 1.0)
  return v * (late & ~st["bad_contact"]).float()


def stand_target(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 per step in the stand conditions (from crouch on), 3 once held for STAND_HOLD_S."""
  ok = stand_ok(env) & (ref_time(env) >= reference(env).kf_t["crouch"])
  held = env.__dict__.get("_getup_held")
  if held is None:
    held = env._getup_held = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
  held[elapsed_steps(env) <= 1] = 0.0
  held[:] = torch.where(ok, held + env.step_dt, torch.zeros_like(held))
  return ok.float() * (1.0 + 2.0 * (held >= STAND_HOLD_S).float())


def face_plant(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = getup_state(env)
  return (st["bad_contact"] & (ref_time(env) >= reference(env).kf_t[PUSH_WINDOW[0]])).float()


def knee_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = getup_state(env)
  return (st["knee_contact"] & (ref_time(env) >= reference(env).kf_t[PUSH_WINDOW[0]])).float()


def reg_scale(env: ManagerBasedRlEnv) -> float:
  t = steps_since_start(env) / (REG_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  return min(1.0, REG_START + (1.0 - REG_START) * t)


def reg_ramped(func: Callable[..., torch.Tensor]) -> Callable:
  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    return func(env, **kwargs) * reg_scale(env)

  return wrapped


def getup_contact_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  return contact_slip_l2(env, sensor_name=GETUP_CONTACT_SENSOR.name)


##
# Terminations.
##


def reference_done(env: ManagerBasedRlEnv) -> torch.Tensor:
  return ref_time(env) > reference(env).t_end + HOLD_S


##
# Environment.
##


def microban_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0)),
    "actions": ObservationTermCfg(func=last_action),
    "getup_phase": ObservationTermCfg(func=getup_phase),
    "ref_joint_pos": ObservationTermCfg(func=ref_joint_pos),
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
  actions = {
    "joint_pos": ReferenceResidualActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=ACTION_SCALE, use_default_offset=False,
      ramp_steps=0,
    ),
  }
  prone_params = {"yaw_range": (-math.pi, math.pi), "tilt_range": (-0.1, 0.1), "joint_pos_range": (-0.1, 0.1)}
  events = {
    "reset_getup": EventTermCfg(
      func=reset_getup, mode="reset",
      params={"p_prone": 1.0 if play else RSI_PRONE, "p_replay": 0.0 if play else RSI_REPLAY,
              "prone_params": prone_params},
    ),
  }
  R = reg_ramped
  rewards = {
    "pose_imitation": RewardTermCfg(func=pose_imitation, weight=1.0),
    "trunk_imitation": RewardTermCfg(func=trunk_imitation, weight=1.0),
    "push_shaping": RewardTermCfg(func=push_shaping, weight=50.0),
    "weight_transfer": RewardTermCfg(func=weight_transfer, weight=2.0),
    "upright_height": RewardTermCfg(func=upright_height, weight=1.0),
    "stand_target": RewardTermCfg(func=stand_target, weight=2.0),
    "face_plant": RewardTermCfg(func=face_plant, weight=-2.0),
    "knee_contact": RewardTermCfg(func=knee_contact, weight=-0.3),
    "torques": RewardTermCfg(func=R(joint_torques_l2), weight=-0.05),
    "action_rate": RewardTermCfg(func=R(action_rate_l2), weight=-0.02),
    "joint_limits": RewardTermCfg(func=R(joint_pos_limits), weight=-5.0),
    "contact_slip": RewardTermCfg(func=R(getup_contact_slip), weight=-1.0),
    "flipped": RewardTermCfg(func=is_terminated, weight=-20.0),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=reference_done, time_out=True),
    "belly_up": TerminationTermCfg(func=belly_up, params={"threshold": 0.3}),
  }
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_microban_prone_robot_cfg()},
      sensors=(GETUP_CONTACT_SENSOR,),
      num_envs=4096,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="trunk",
      distance=0.9, elevation=-15.0, azimuth=90.0,
    ),
    sim=SimulationCfg(nconmax=None, njmax=None,
                      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4,
    episode_length_s=20.0,  # safety; episodes end by reference_done
  )
  if play:
    cfg.scene.num_envs = 16
  return cfg


def microban_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=SymmetricPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="microban_getup",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=4000,
  )
