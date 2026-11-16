"""Run walk.onnx in MJLab from a standing start and report, per environment, whether it walks.

Each env stands still (command 0) for --settle-s, then gets the forward command
--vx for --seconds. A fall = trunk tilt above 60 deg (projected gravity z > -0.5,
microban walk.py's own fall check) or trunk height below 0.10 m, latched.

--sweep randomizes the start pose per env (roll/pitch, joint offsets from the
default pose, trunk velocity, a small drop gap; the pose is placed with its lowest
collision point at the floor + gap) and prints the success rate as a function of
each perturbation -- the walk.onnx "start pose conditions".

  uv run python scripts/walk_onnx_check.py --num-envs 256 --vx 0.3
  uv run python scripts/walk_onnx_check.py --num-envs 4096 --sweep
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.robot import INIT_HEIGHT
from microban_rl.walk_onnx import KP_RL_STIFFNESS, WALK_JOINTS, WalkOnnxPolicy, walk_onnx_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

FALL_GRAVITY_Z = -0.5
FALL_HEIGHT = 0.10


@dataclass(frozen=True)
class Args:
  num_envs: int = 256
  vx: float = 0.3
  vy: float = 0.0
  vtheta: float = 0.0
  settle_s: float = 1.0
  seconds: float = 10.0
  kp: float = KP_RL_STIFFNESS
  delays: bool = True
  sweep: bool = False
  """Randomize the start pose (see SWEEP) and report success vs. perturbation."""
  seed: int = 0
  out: str = ""
  roll_max: float = 0.5
  pitch_range: tuple[float, float] = (-0.5, 0.5)
  joint_max: float = 0.4
  lin_vel_max: float = 0.3
  ang_vel_max: float = 1.5
  gap_max: float = 0.02




def lowest_point(env) -> torch.Tensor:
  """Lowest point of the robot's collision geoms (boxes, cylinders) above z = 0. [B]"""
  import mujoco

  mj = env.sim.mj_model
  g = [i for i in range(mj.ngeom) if mj.geom_contype[i] or mj.geom_conaffinity[i]]
  box_t, cyl_t = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)
  g = [i for i in g if int(mj.geom_type[i]) in (box_t, cyl_t)]
  dev = env.device
  idx = torch.tensor(g, device=dev, dtype=torch.long)
  size = torch.tensor(mj.geom_size[g], device=dev, dtype=torch.float32)
  is_box = torch.tensor([int(mj.geom_type[i]) == box_t for i in g], device=dev, dtype=torch.bool)
  pos = env.sim.data.geom_xpos[:, idx]  # [B, G, 3]
  rz = env.sim.data.geom_xmat[:, idx].reshape(pos.shape[0], len(g), 3, 3)[:, :, 2, :]  # z row
  box = pos[..., 2] - (rz.abs() * size).sum(-1)
  cz = rz[..., 2].abs()
  cyl = pos[..., 2] - cz * size[:, 1] - size[:, 0] * (1 - cz**2).clamp(min=0).sqrt()
  return torch.where(is_box, box, cyl).amin(1)


def main(a: Args) -> None:
  device = "cuda:0"
  torch.manual_seed(a.seed)
  cfg = walk_onnx_env_cfg(a.num_envs, kp=a.kp)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  robot = env.scene["robot"]
  n = a.num_envs
  names = list(robot.joint_names)
  walk_ids = [names.index(j) for j in WALK_JOINTS]

  pert = {}
  if a.sweep:
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(n, *s, device=device)  # noqa: E731
    # Uniform ranges (Args *_max); joint offsets are scaled per env by a factor in [0, 1].
    pert["gap"] = u(0.0, a.gap_max)
    pert["roll"], pert["pitch"] = u(-a.roll_max, a.roll_max), u(*a.pitch_range)
    scale = torch.rand(n, 1, device=device)
    joint = u(-a.joint_max, a.joint_max, len(names)) * scale
    joint[:, names.index("head")] = 0.0
    pert["joint_max"] = joint.abs().amax(1)
    pert["joint_rms"] = joint[:, walk_ids].square().mean(1).sqrt()
    lin = u(-a.lin_vel_max, a.lin_vel_max, 3)
    ang = u(-a.ang_vel_max, a.ang_vel_max, 3)
    pert["lin_vel"], pert["ang_vel"] = lin.norm(dim=1), ang.norm(dim=1)
    root = robot.data.default_root_state.clone()
    root[:, 2] = INIT_HEIGHT
    root[:, 3:7] = quat_from_euler_xyz(pert["roll"], pert["pitch"], torch.zeros_like(pert["roll"]))
    root[:, 7:10], root[:, 10:13] = lin, ang
    root[:, :2] += env.scene.env_origins[:, :2]
    q = robot.data.default_joint_pos + joint
    lim = robot.data.soft_joint_pos_limits
    q = q.clamp(lim[..., 0], lim[..., 1])
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.sim.forward()
    root[:, 2] += pert["gap"] - lowest_point(env)
    pert["trunk_z"] = root[:, 2].clone()
    robot.write_root_state_to_sim(root)
    env.sim.forward()

  policy = WalkOnnxPolicy(robot, n, device, delays=a.delays)
  cmd = torch.zeros(n, 3, device=device)
  settle = int(a.settle_s / env.step_dt)
  steps = settle + int(a.seconds / env.step_dt)
  fallen = torch.zeros(n, dtype=torch.bool, device=device)
  fall_t = torch.full((n,), float("nan"), device=device)
  pos0 = None
  head0 = None
  vmax = torch.zeros(n, device=device)
  tau_max = torch.zeros(n, device=device)
  with torch.inference_mode():
    for t in range(steps):
      if t == settle:
        cmd[:] = torch.tensor([a.vx, a.vy, a.vtheta], device=device)
        pos0 = robot.data.root_link_pos_w[:, :2].clone()
        head0 = quat_apply(robot.data.root_link_quat_w, torch.tensor([[1.0, 0, 0]], device=device).expand(n, 3))
      env.step(policy.act(cmd))
      d = robot.data
      bad = (d.projected_gravity_b[:, 2] > FALL_GRAVITY_Z) | (d.root_link_pos_w[:, 2] < FALL_HEIGHT)
      fall_t = torch.where(bad & ~fallen, torch.full_like(fall_t, t * env.step_dt), fall_t)
      fallen |= bad
      if t >= settle:
        vmax = torch.maximum(vmax, d.joint_vel.abs().amax(1))
        tau_max = torch.maximum(tau_max, d.actuator_force.abs().amax(1))
    disp = robot.data.root_link_pos_w[:, :2] - pos0
    head1 = quat_apply(robot.data.root_link_quat_w, torch.tensor([[1.0, 0, 0]], device=device).expand(n, 3))
  # Direction of travel vs. the trunk's x axis at the start of walking.
  h0 = head0[:, :2] / head0[:, :2].norm(dim=1, keepdim=True)
  fwd = (disp * h0).sum(1)
  lat = h0[:, 0] * disp[:, 1] - h0[:, 1] * disp[:, 0]
  yaw = torch.atan2(head1[:, 1], head1[:, 0]) - torch.atan2(head0[:, 1], head0[:, 0])
  yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
  ok = ~fallen
  r = {
    "kp": a.kp, "delays": a.delays, "vx": a.vx, "sweep": a.sweep, "num_envs": n,
    "success_rate": float(ok.float().mean()),
    "fall_time_median_s": float(np.nanmedian(fall_t.cpu().numpy())) if fallen.any() else None,
    "fwd_speed_median": float((fwd[ok] / a.seconds).median()) if ok.any() else None,
    "fwd_speed_p10": float((fwd[ok] / a.seconds).quantile(0.1)) if ok.any() else None,
    "lateral_m_median": float(lat[ok].median()) if ok.any() else None,
    "yaw_change_deg_median": float(torch.rad2deg(yaw[ok]).median()) if ok.any() else None,
    "joint_vel_max": float(vmax.max()), "torque_max": float(tau_max.max()),
  }
  print("RESULT " + json.dumps(r), flush=True)

  if a.sweep:
    okn = ok.cpu().numpy()
    print("success rate vs. start perturbation (bins of equal width):")
    for k, v in pert.items():
      v = v.cpu().numpy()
      if k in ("roll", "pitch", "gap", "trunk_z"):
        edges = np.linspace(v.min(), v.max(), 9)
      else:
        edges = np.linspace(0, v.max(), 9)
      idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
      row = [f"{(edges[i] + edges[i + 1]) / 2:+.3f}:{okn[idx == i].mean() * 100:3.0f}%" for i in range(len(edges) - 1)]
      print(f"  {k:9s} " + "  ".join(row))
    if a.out:
      np.savez(a.out, ok=okn, **{k: v.cpu().numpy() for k, v in pert.items()})


if __name__ == "__main__":
  main(tyro.cli(Args))
