"""Export the final Microban policies to ONNX with walk.onnx-style metadata.

For each (task, checkpoint): the actor MLP with its observation normalizer
(rsl-rl's ONNX wrapper; deterministic output = the Gaussian mean), input "obs"
[1, D], output "actions" [1, 19]. Metadata (onnx metadata_props, all strings):

  task, checkpoint, joint_names (= action / joint_pos / joint_vel order),
  observation_names, observation_dims, observation_scales (per term),
  observation_description, default_joint_pos, action_scale, action_rule,
  control_dt, kp, kd, torque_limit, target_speed_limit, plus task extras.

Each export is checked against the torch policy on observations recorded
from play rollouts (max |torch - onnxruntime|).

  uv run python scripts/export_onnx.py --out ~/mero/onnx
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.robot import DEFAULT_JOINT_POS, XL330_DAMPING, XL330_EFFORT_LIMIT, XL330_STIFFNESS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

LOGS = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"
POLICIES = {
  "microban_stand": ("Microban-Stand", LOGS / "microban_stand/2026-10-30_11-37-20/model_299.pt"),
  "microban_crawl_arms2": ("Microban-Crawl-Arms-2", LOGS / "microban_crawl_arms/2026-11-04_11-43-11/model_5497.pt"),
  "microban_getup5": ("Microban-GetUp-5", LOGS / "microban_getup/2026-11-16_21-07-16_getup5/model_2500.pt"),
}
OBS_DESCRIPTION = {
  "base_ang_vel": "trunk angular velocity in the trunk frame (rad/s); on the robot: gyro rotated from the IMU frame by IMU_MOUNT_QUAT (0.5,-0.5,-0.5,0.5)",
  "projected_gravity": "unit gravity (0,0,-1) expressed in the trunk frame; on the robot: from the IMU orientation estimate",
  "heading": "(sin, cos) of the angle between the head direction (trunk +z) projected on the ground and the crawl goal direction (world +x at the start); on the robot: IMU yaw relative to its value at the start",
  "joint_pos": "joint position - default_joint_pos (rad), joint_names order",
  "joint_vel": "joint velocity (rad/s), joint_names order",
  "actions": "the policy's previous raw output (before scaling), joint_names order; zeros at the start",
  "getup_phase": "clip(t_ref / t_end, 0, 1), t_ref = seconds since the get-up start - 0.5 (t_ref = -0.5 at the start, prone settle), t_end = 12.4",
  "ref_joint_pos": "q_ref(t_ref + control_dt) - default_joint_pos (rad), from getup_reference.npz (q_ref row = round(t / 0.02), clamped to the last row)",
}


@dataclass(frozen=True)
class Args:
  out: str = str(Path.home() / "mero/onnx")
  num_envs: int = 8
  check_steps: int = 300


def main(a: Args) -> None:
  out = Path(a.out).expanduser()
  out.mkdir(parents=True, exist_ok=True)
  summary = {}
  for name, (task, ckpt) in POLICIES.items():
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = a.num_envs
    ag = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(ve, asdict(ag), device=env.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    runner.export_policy_to_onnx(str(out), f"{name}.onnx")
    path = out / f"{name}.onnx"

    robot = env.scene["robot"]
    joints = [n.split("/")[-1] for n in robot.joint_names]
    term = env.action_manager.get_term("joint_pos")
    targets = [n.split("/")[-1] for n in term.target_names]
    assert targets == joints, (targets, joints)
    om = env.observation_manager
    obs_names = list(om.active_terms["actor"])
    obs_dims = [int(d[-1]) for d in om.group_obs_term_dim["actor"]]
    obs_scales = []
    for t in obs_names:
      sc = om.get_term_cfg("actor", t).scale
      obs_scales.append(1.0 if sc is None else float(sc))
    scale = float(term.cfg.scale)
    speed = getattr(term.cfg, "max_speed", None)
    if task == "Microban-GetUp-5":
      rule = ("target = q_ref(t_ref + control_dt) + action_scale * action, where inside the push window "
              "(t_ref in [8.0, 10.4] s) the residual of hip_yaw, hip_roll, ankle_roll is clipped to +-0.1 rad; "
              "then target - previous target is clipped to +-target_speed_limit * control_dt")
    elif speed is not None:
      rule = ("target = default_joint_pos + action_scale * action; then target - previous target is clipped to "
              "+-target_speed_limit * control_dt (first step: from the measured joint positions)")
    else:
      rule = "target = default_joint_pos + action_scale * action"
    meta = {
      "task": task, "checkpoint": str(ckpt.relative_to(LOGS.parents[1])),
      "joint_names": ",".join(joints),
      "observation_names": ",".join(obs_names),
      "observation_dims": ",".join(map(str, obs_dims)),
      "observation_scales": ",".join(f"{s:g}" for s in obs_scales),
      "observation_description": json.dumps({t: OBS_DESCRIPTION[t] for t in obs_names}),
      "default_joint_pos": ",".join(f"{DEFAULT_JOINT_POS[j]:.4f}" for j in joints),
      "action_scale": f"{scale:g}",
      "action_rule": rule,
      "control_dt": f"{env.step_dt:g}",
      "kp": f"{XL330_STIFFNESS:g}", "kd": f"{XL330_DAMPING:g}",
      "torque_limit": f"{XL330_EFFORT_LIMIT:.4f}",
      "target_speed_limit": "none" if speed is None else f"{float(speed):.3f}",
      "position_actuator": "tau = kp * (target - q) - kd * qdot, |tau| <= torque_limit (Nm, rad)",
    }
    if task == "Microban-GetUp-5":
      meta.update({"reference_file": "getup_reference.npz", "reference_t_end": "12.4", "reference_t_start": "-0.5",
                   "push_window": "8.0,10.4", "push_clip_joints": "hip_yaw,hip_roll,ankle_roll", "push_clip_rad": "0.1"})
    m = onnx.load(str(path))
    del m.metadata_props[:]
    for k, v in meta.items():
      p = m.metadata_props.add()
      p.key, p.value = k, v
    onnx.save(m, str(path))

    # torch vs onnxruntime on recorded observations
    sess = ort.InferenceSession(str(path))
    iname = sess.get_inputs()[0].name
    errs = []
    with torch.inference_mode():
      env.reset()
      obs = ve.get_observations()
      for s in range(a.check_steps):
        act = policy(obs)
        if s % 10 == 0:
          x = obs["actor"].cpu().numpy().astype(np.float32)
          for i in range(x.shape[0]):
            y = sess.run(None, {iname: x[i : i + 1]})[0][0]
            errs.append(float(np.abs(y - act[i].cpu().numpy()).max()))
        obs, _, _, _ = ve.step(act)
    summary[name] = {"task": task, "obs_dim": int(sum(obs_dims)), "obs": dict(zip(obs_names, obs_dims)),
                     "action_scale": scale, "target_speed_limit": meta["target_speed_limit"],
                     "max_abs_diff": max(errs), "compared": len(errs)}
    print(f"{name}: {path} obs {sum(obs_dims)} = {dict(zip(obs_names, obs_dims))}, scale {scale}, "
          f"speed limit {meta['target_speed_limit']} | torch vs onnxruntime max |diff| {max(errs):.2e} "
          f"over {len(errs)} observations", flush=True)
    env.close()
  ref = Path(__file__).resolve().parents[1] / "data" / "getup_reference.npz"
  shutil.copy(ref, out / "getup_reference.npz")
  print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
  main(tyro.cli(Args))
