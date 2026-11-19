"""Evaluate a Microban-Walk checkpoint and export it to ONNX for mero_walk.

The ONNX uses walk.onnx's interface (mero_walk/mero/policy.py): input "obs"
[1, 63], output "actions" [1, 18], metadata joint_names (18, no head),
default_joint_pos, action_scale. Run it on the robot with

  mero run --policy models/<name>.onnx --gyro-frame body

Evaluation (play config: no pushes; floor DR, backlash, gains, delays still
randomized per env): each env gets a fixed command for --seconds; reports the
fall rate and velocity tracking per command.

  uv run python scripts/export_walk_onnx.py logs/rsl_rl/microban_walk/<run>/model_2999.pt \
      --out ../mero_walk/models/mero_walk.onnx
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.robot import DEFAULT_JOINT_POS
from microban_rl.walk_env_cfg import ACTION_SCALE, CONTROL_HZ, KP_WALK_REGISTER_5V
from microban_rl.walk_onnx import KP_RL_STIFFNESS, WALK_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = os.environ.get("TASK", "Microban-Walk")
# (vx, vy, wz) evaluation commands, one group of envs each.
EVAL_COMMANDS = [
  (0.0, 0.0, 0.0), (0.10, 0.0, 0.0), (0.20, 0.0, 0.0), (0.30, 0.0, 0.0),
  (-0.15, 0.0, 0.0), (0.0, 0.10, 0.0), (0.0, 0.0, 0.5), (0.15, 0.0, -0.4),
]


@dataclass(frozen=True)
class Args:
  checkpoint: tyro.conf.Positional[str]
  out: str = ""
  envs_per_command: int = 32
  seconds: float = 10.0
  push: bool = False
  """Keep the training pushes (+-0.5 m/s every 1-3 s)."""
  hard: bool = False
  """Worst-case DR: backlash 3 deg, kp and torque limit x0.75..0.8, floor friction 0.4..0.5 (Walk-DR only)."""


def main(a: Args) -> None:
  ckpt = Path(a.checkpoint).resolve()
  cfg = load_env_cfg(TASK, play=True)
  n = a.envs_per_command * len(EVAL_COMMANDS)
  cfg.scene.num_envs = n
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  if a.push:
    cfg.events["push_robot"] = load_env_cfg(TASK).events["push_robot"]
  if a.hard:
    act = cfg.scene.entities["robot"].articulation.actuators[0]
    act.backlash_range = (math.radians(3.0), math.radians(3.0))
    act.kp_scale_range = (0.75, 0.8)
    act.effort_scale_range = (0.75, 0.8)
    cfg.events["floor_friction"].params["ranges"] = (0.4, 0.5)
  ag = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  runner = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)

  # --- evaluation ---
  cmd = torch.tensor(EVAL_COMMANDS, device=env.device).repeat_interleave(a.envs_per_command, 0)
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  env.reset()
  term.vel_command_b[:] = cmd
  term.is_standing_env[:] = False
  obs = ve.get_observations()
  steps = int(a.seconds / env.step_dt)
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  vel_sum = torch.zeros(n, 3, device=env.device)
  cnt = torch.zeros(n, device=env.device)
  half = steps // 3
  record: list[np.ndarray] = []
  record_act: list[np.ndarray] = []
  with torch.inference_mode():
    for s in range(steps):
      term.vel_command_b[:] = cmd
      act = policy(obs)
      if s % 15 == 0:
        record.append(obs["actor"][:4].cpu().numpy())
        record_act.append(act[:4].cpu().numpy())
      obs, _, dones, _ = ve.step(act)
      fell |= dones.bool() & ~env.termination_manager.time_outs
      if s >= half:
        ok = ~fell
        v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], 1)
        vel_sum[ok] += v[ok]
        cnt[ok] += 1
  mean_v = vel_sum / cnt.clamp(min=1)[:, None]
  print(f"\n{'command (vx,vy,wz)':>22} | fall % | measured vx, vy, wz (mean over the last 2/3)")
  report = []
  for i, c in enumerate(EVAL_COMMANDS):
    sl = slice(i * a.envs_per_command, (i + 1) * a.envs_per_command)
    f = float(fell[sl].float().mean() * 100)
    mv = mean_v[sl][~fell[sl]].mean(0).tolist() if (~fell[sl]).any() else [float("nan")] * 3
    print(f"{str(c):>22} | {f:6.1f} | {mv[0]:+.3f} {mv[1]:+.3f} {mv[2]:+.3f}")
    report.append({"command": c, "fall_pct": f, "measured": mv})
  print("overall fall %:", float(fell.float().mean() * 100))

  # --- export ---
  if a.out:
    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(out.parent), out.name)
    targets = [t.split("/")[-1] for t in env.action_manager.get_term("joint_pos").target_names]
    assert targets == list(WALK_JOINTS), targets
    meta = {
      "task": TASK, "checkpoint": str(ckpt),
      "joint_names": ",".join(WALK_JOINTS),
      "default_joint_pos": ",".join(f"{DEFAULT_JOINT_POS[j]:.4f}" for j in WALK_JOINTS),
      "action_scale": f"{ACTION_SCALE:g}",
      "action_rule": "target = default_joint_pos + action_scale * action (head held at 0)",
      "observation": "base_ang_vel(3, trunk frame) projected_gravity(3) joint_pos-default(18) joint_vel(18) "
                     "last_action(18) command vx,vy,wz(3)",
      "gyro_frame": "body",
      "control_hz": str(CONTROL_HZ),
      "kp_walk_register_5v": str(KP_WALK_REGISTER_5V),
      "kp_nm_per_rad": f"{KP_RL_STIFFNESS:g}",
      "eval": json.dumps(report),
    }
    m = onnx.load(str(out))
    del m.metadata_props[:]
    for k, v in meta.items():
      p = m.metadata_props.add()
      p.key, p.value = k, v
    onnx.save(m, str(out))
    sess = ort.InferenceSession(str(out))
    iname = sess.get_inputs()[0].name
    x = np.concatenate(record).astype(np.float32)
    y_torch = np.concatenate(record_act)
    y_onnx = np.stack([sess.run(None, {iname: row[None]})[0][0] for row in x])
    err = float(np.abs(y_onnx - y_torch).max())
    print(f"exported {out}  in {sess.get_inputs()[0].shape} out {sess.get_outputs()[0].shape}  "
          f"torch vs onnxruntime max |diff| {err:.2e} over {len(x)} observations")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
