"""The microban repo's walk.onnx policy, run inside MJLab.

walk.onnx (microban/src/agents, trained by Rhoban, run path 2026-06-26_16-03-28)
is an MLP 63 -> 512 -> 256 -> 128 -> 18 (ELU) with a built-in observation
normalizer; its weights are loaded into torch here so it runs batched on the GPU.
Its interface, from microban/src/moves/walk.py, constants.py and
sim/mujoco_controller.py:

- 50 Hz, observation (63):
    gyro (3)               angular velocity in the IMU frame (the "imu" site,
                           quat IMU_MOUNT_QUAT relative to the trunk), rad/s
    projected gravity (3)  world (0, 0, -1) in the trunk frame
    joint pos (18)         q - default, WALK_JOINTS order (no head)
    joint vel (18)         dq, rad/s
    last action (18)
    command (3)            vx, vy (m/s), vtheta (rad/s)
- action (18): joint target = default + 1.0 * action.
- default pose = the ONNX default_joint_pos metadata (= robot.DEFAULT_JOINT_POS).
- motor gain while walking: KP_RL = 125 register units (~0.275 Nm/rad), instead
  of the KP_DEFAULT = 400 (0.88 Nm/rad) robot.py uses; firmware current limit
  1.75 A (0.64 Nm) as in robot.py.
- microban's own simulator adds sensor delays (velocity 1, gyro 3, projected
  gravity 4 ticks at 50 Hz; actuation 2 physics steps) and the BAM XL330 motor
  model; WalkOnnxPolicy reproduces the tick delays (delays=True).
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from microban_rl.robot import (
  DEFAULT_JOINT_POS,
  MICROBAN_ARTICULATION,
  MICROBAN_ROOT,
  XL330_ACTUATOR_CFG,
)

WALK_ONNX = MICROBAN_ROOT / "src" / "agents" / "walk.onnx"
# microban constants.OBSERVATION_DOF_ORDER (= MJCF joint order without the head).
WALK_JOINTS = (
  "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
  "right_ankle_pitch", "right_ankle_roll",
  "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
  "left_ankle_pitch", "left_ankle_roll",
)
IMU_MOUNT_QUAT = (0.5, -0.5, -0.5, 0.5)  # (w, x, y, z), constants.py / robot.xml site "imu"
KP_RL_STIFFNESS = 125 * 0.0022  # Nm/rad
# microban sim_main.py defaults, in 50 Hz ticks.
DELAY_TICKS = {"vel": 1, "gyro": 3, "gravity": 4}

# Same motors, joint ranges and torque limit as robot.py; only the P gain differs.
WALK_ARTICULATION = replace(
  MICROBAN_ARTICULATION,
  actuators=(replace(XL330_ACTUATOR_CFG, stiffness=KP_RL_STIFFNESS),),
)


def load_walk_mlp(path: str | Path = WALK_ONNX, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
  """walk.onnx as a torch module (normalizer included) + its metadata."""
  import onnx
  from onnx import numpy_helper

  m = onnx.load(str(path))
  w = {t.name: torch.tensor(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
  meta = {p.key: p.value for p in m.metadata_props}

  class WalkMlp(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.register_buffer("mean", w["obs_normalizer._mean"])
      self.register_buffer("std", w["onnx::Div_24"])
      layers = []
      for i in (0, 2, 4, 6):
        lin = torch.nn.Linear(*w[f"mlp.{i}.weight"].shape[::-1])
        lin.weight.data, lin.bias.data = w[f"mlp.{i}.weight"], w[f"mlp.{i}.bias"]
        layers += [lin, torch.nn.ELU()]
      self.mlp = torch.nn.Sequential(*layers[:-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      return self.mlp((obs - self.mean) / self.std)

  names = meta["joint_names"].split(",")
  default = dict(zip(names, (float(v) for v in meta["default_joint_pos"].split(","))))
  for n in WALK_JOINTS:
    assert abs(default[n] - DEFAULT_JOINT_POS[n]) < 1e-3, n
  return WalkMlp().to(device).eval(), meta


class WalkOnnxPolicy:
  """Builds walk.onnx observations from an MJLab robot entity and returns targets.

  act(command) -> [B, 19] joint position *offsets* from the default pose in
  the entity's joint order (head held at 0), for a JointPositionAction with
  scale 1 and use_default_offset.
  """

  def __init__(self, robot, num_envs: int, device: str, delays: bool = True):
    self.robot = robot
    self.mlp, self.meta = load_walk_mlp(device=device)
    names = list(robot.joint_names)
    self.ids = torch.tensor([names.index(n) for n in WALK_JOINTS], device=device)
    self.default = torch.tensor([DEFAULT_JOINT_POS[n] for n in WALK_JOINTS], device=device)
    self.mount = torch.tensor(IMU_MOUNT_QUAT, device=device).expand(num_envs, 4)
    self.last_action = torch.zeros(num_envs, len(WALK_JOINTS), device=device)
    self.n_joints = len(names)
    self.delays = {k: (v if delays else 0) for k, v in DELAY_TICKS.items()}
    self._buf: dict[str, deque] = {}

  def _delayed(self, key: str, x: torch.Tensor, env_ids=None) -> torch.Tensor:
    n = self.delays.get(key, 0)
    buf = self._buf.get(key)
    if buf is None:
      buf = deque([x.clone()] * (n + 1), maxlen=n + 1)
      self._buf[key] = buf
    buf.append(x.clone())
    return buf[0]

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear last action and fill the delay buffers with the current state."""
    if env_ids is None:
      self.last_action.zero_()
      self._buf.clear()
      return
    self.last_action[env_ids] = 0.0
    cur = self._raw()
    for k, buf in self._buf.items():
      for x in buf:
        x[env_ids] = cur[k][env_ids]

  def _raw(self) -> dict[str, torch.Tensor]:
    d = self.robot.data
    return {
      "gyro": quat_apply_inverse(self.mount, d.root_link_ang_vel_b),
      "gravity": d.projected_gravity_b,
      "pos": d.joint_pos[:, self.ids] - self.default,
      "vel": d.joint_vel[:, self.ids],
    }

  def observation(self, command: torch.Tensor) -> torch.Tensor:
    raw = self._raw()
    obs = [self._delayed(k, raw[k]) for k in ("gyro", "gravity", "pos", "vel")]
    return torch.cat([*obs, self.last_action, command], dim=1)

  @torch.no_grad()
  def act(self, command: torch.Tensor) -> torch.Tensor:
    action = self.mlp(self.observation(command))
    self.last_action = action
    out = torch.zeros(action.shape[0], self.n_joints, device=action.device)
    out[:, self.ids] = action
    return out


def onnx_reference(obs: np.ndarray) -> np.ndarray:
  """walk.onnx through onnxruntime (batch 1), for checking the torch copy."""
  import onnxruntime as ort

  s = ort.InferenceSession(str(WALK_ONNX))
  return np.stack([s.run(None, {"obs": o[None].astype(np.float32)})[0][0] for o in obs])


def walk_onnx_env_cfg(num_envs: int = 64, kp: float = KP_RL_STIFFNESS):
  """Play-only env for running walk.onnx: standing start, no pushes, no early resets.

  Microban-Stand's play config with the walk P gain (kp), action scale 1 around
  the default pose, and only the time-out termination (falls are detected by
  the caller). Joint ranges, torque limit and the MJCF are those of robot.py.
  """
  from mjlab.envs.mdp.actions import JointPositionActionCfg

  from microban_rl.robot import get_microban_robot_cfg
  from microban_rl.stand_env_cfg import microban_stand_env_cfg

  cfg = microban_stand_env_cfg(play=True)
  robot = get_microban_robot_cfg()
  robot.articulation = replace(
    MICROBAN_ARTICULATION, actuators=(replace(XL330_ACTUATOR_CFG, stiffness=kp),)
  )
  cfg.scene.entities["robot"] = robot
  cfg.scene.num_envs = num_envs
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=True
    )
  }
  # Reset to the robot.py initial state (default pose, INIT_HEIGHT, facing +x),
  # no randomization and no pushes.
  cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_joints")}
  cfg.events["reset_base"].params = {"pose_range": {}, "velocity_range": {}}
  cfg.events["reset_joints"].params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
  cfg.episode_length_s = 1e9
  return cfg
