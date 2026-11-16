"""Per-environment checks of a Microban-GetUp run (or one checkpoint).

Every env starts prone (play config: prone.py randomization, random heading)
and runs one full episode (reference + HOLD_S); an env that flips onto its back
is counted up to that moment. Nothing is averaged across envs before taking a
magnitude. Reported:

- last stage passed (furthest stage whose condition held at some moment; with
  --sequential each stage also needs the previous one, as in runs 1-3):
    pike_1     hands and feet down, trunk tilt >= 100 deg, trunk z >= 0.13
    pike_push  hands and feet down, CoM within 10 mm of the toes, z >= 0.13
    fold       feet only (no hands / trunk), capture point inside the soles, z >= 0.11
    squat_deep feet only, capture point inside the soles, tilt <= 45 deg
    crouch     feet only, capture point inside the soles, tilt <= 30 deg, z >= 0.10
    stand      the stand conditions (stage 1) held for 1 s  (= success)
- success rate, get-up time (start of the 1 s hold), flip rate
- posture asymmetry per env: mean over joint pairs |mean_t q_L - sign mean_t q_R|,
  over the whole episode and per segment (early: until pike_1, push: pike_1 ->
  squat_deep, late: after squat_deep), with the worst joint pair of the push segment
- peak torque and joint speed per env
- walk.onnx start conditions (|roll| < 0.2, pitch -0.1..+0.5, joint RMS < 0.12,
  both soles down, nearly still) at the last keyframe time, 1 s after it and at
  the end of the episode.

Watch mode (run dir): evaluates model_<k>.pt every --every iterations and stops
--train-pid (SIGINT) when the asymmetry median exceeds --stop-asym (whole
episode) or --stop-push-asym (push segment), or when the mean stage and the
success rate have not improved for STALL_EVALS evaluations.

  uv run python scripts/monitor_getup.py logs/rsl_rl/microban_getup/<run> --train-pid <pid>
  uv run python scripts/monitor_getup.py --checkpoint-file <model.pt>
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.getup_env_cfg import HOLD_S, STAND_HOLD_S, getup_state, reference, ref_time, stand_ok
from microban_rl.symmetry import _MIRROR_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

STAGES = ("none", "pike_1", "pike_push", "fold", "squat_deep", "crouch", "stand")
STOP_ASYM = 0.10  # rad
STALL_EVALS = 2
MIN_GAIN_STAGE, MIN_GAIN_SUCCESS = 0.05, 0.02


@dataclass(frozen=True)
class Args:
  run_dir: tyro.conf.Positional[str] = ""
  checkpoint_file: str = ""
  task_id: str = "Microban-GetUp"
  train_pid: int | None = None
  first: int = 500
  every: int = 500
  last: int = 4000
  num_envs: int = 512
  seed: int = 1
  stop_asym: float = STOP_ASYM
  stop_push_asym: float = float("inf")
  sequential: bool = False
  """Stages must be passed in order (the original definition; a hop past pike_push then never counts)."""
  min_success: float = 0.0
  """Stop when the success rate falls below this (after the first evaluation at or above it)."""
  stall_below_success: float = 1.01
  """Count 'no progress' only while the success rate is below this."""


def walk_start_ok(env) -> torch.Tensor:
  st = getup_state(env)
  robot = env.scene["robot"]
  d = robot.data
  ref = reference(env)
  names = [n.split("/")[-1] for n in robot.joint_names]
  walk = torch.tensor([i for i, n in enumerate(names) if n != "head"], device=env.device)
  rms = (d.joint_pos[:, walk] - ref.default[walk]).square().mean(1).sqrt()
  return ((st["roll"].abs() < 0.2) & (st["pitch"] > -0.1) & (st["pitch"] < 0.5) & (rms < 0.12)
          & st["feet_down"] & st["soles_flat"] & ~st["hands_down"] & ~st["bad_contact"]
          & (d.joint_vel.abs().amax(1) < 1.0) & (d.root_link_lin_vel_w.norm(dim=-1) < 0.1)
          & (d.root_link_ang_vel_w.norm(dim=-1) < 0.5))


def stage_conditions(env) -> list[torch.Tensor]:
  st = getup_state(env)
  inside = (st["cp"] <= st["toe"]) & (st["cp"] >= st["heel"])
  feet_only = st["feet_down"] & ~st["hands_down"] & ~st["bad_contact"]
  four = st["hands_down"] & st["feet_down"]
  return [
    four & (st["tilt"] >= math.radians(100)) & (st["z"] >= 0.13),
    four & (st["com_x"] - st["toe"] <= 0.010) & (st["z"] >= 0.13),
    feet_only & inside & (st["z"] >= 0.11),
    feet_only & inside & (st["tilt"] <= math.radians(45)),
    feet_only & inside & (st["tilt"] <= math.radians(30)) & (st["z"] >= 0.10),
  ]


def evaluate(env, vec_env, agent_cfg, ckpt: str, seed: int, sequential: bool = False) -> dict:
  torch.manual_seed(seed)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=env.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)
  robot = env.scene["robot"]
  names = [n.split("/")[-1] for n in robot.joint_names]
  n, dev = env.num_envs, env.device
  with torch.inference_mode():
    env.reset()
    obs = vec_env.get_observations()
  ref = reference(env)
  steps = int(round((ref.t_end + HOLD_S - float(env._getup_t0[0])) / env.step_dt))
  stage = torch.zeros(n, dtype=torch.long, device=dev)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  flipped = torch.zeros(n, dtype=torch.bool, device=dev)
  held = torch.zeros(n, device=dev)
  t_up = torch.full((n,), float("nan"), device=dev)
  marks = {"t_end": ref.t_end, "t_end+1s": ref.t_end + 1.0}
  walk_at = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in (*marks, "episode_end", "after_getup")}
  qsum = torch.zeros(n, len(names), device=dev)
  cnt = torch.zeros(n, device=dev)
  seg_bounds = {"early": (-1e9, ref.kf_t["pike_1"]), "push": (ref.kf_t["pike_1"], ref.kf_t["squat_deep"]),
                "late": (ref.kf_t["squat_deep"], 1e9)}
  seg_q = {k: torch.zeros(n, len(names), device=dev) for k in seg_bounds}
  seg_c = {k: torch.zeros(n, device=dev) for k in seg_bounds}
  tau_max = torch.zeros(n, device=dev)
  vel_max = torch.zeros(n, device=dev)
  with torch.inference_mode():
    for s in range(steps):
      obs, _, dones, extras = vec_env.step(policy(obs))
      done = dones.bool()
      term = done & ~extras["time_outs"].bool() if "time_outs" in extras else done
      # the state after a done is the next episode's start: use flags from before the reset
      flipped |= term & alive
      alive &= ~done
      if s == steps - 1:
        alive_end = alive.clone()
      if not alive.any():
        break
      d = robot.data
      a = alive.float()
      qsum += (d.joint_pos - ref.default) * a[:, None]
      cnt += a
      tr = ref_time(env)
      for k, (lo, hi) in seg_bounds.items():
        m_ = a * ((tr >= lo) & (tr < hi)).float()
        seg_q[k] += (d.joint_pos - ref.default) * m_[:, None]
        seg_c[k] += m_
      tau_max = torch.where(alive, torch.maximum(tau_max, d.actuator_force.abs().amax(1)), tau_max)
      vel_max = torch.where(alive, torch.maximum(vel_max, d.joint_vel.abs().amax(1)), vel_max)
      conds = stage_conditions(env)
      for k, c in enumerate(conds, start=1):
        prev = (stage == k - 1) if sequential else (stage < k)
        stage = torch.where(alive & prev & c, torch.full_like(stage, k), stage)
      ok = stand_ok(env) & alive & ((stage >= 5) if sequential else True)
      held = torch.where(ok, held + env.step_dt, torch.zeros_like(held))
      just = (held >= STAND_HOLD_S) & ((stage == 5) if sequential else (stage < 6))
      t_up = torch.where(just, ref_time(env) - STAND_HOLD_S, t_up)
      w = walk_start_ok(env) & alive
      walk_at["after_getup"] |= just & w
      stage = torch.where(just, torch.full_like(stage, 6), stage)
      t = ref_time(env)
      for k, tm in marks.items():
        hit = (t - tm).abs() < env.step_dt / 2
        walk_at[k] = torch.where(hit & alive, w, walk_at[k])
      if s == steps - 2:
        walk_at["episode_end"] = w
        pitch_end = getup_state(env)["pitch"].clone()
  st_np = stage.cpu().numpy()
  mean_q = (qsum / cnt.clamp(min=1)[:, None]).cpu().numpy()
  pairs = [(names.index(l), names.index(r), sgn) for l, (r, sgn) in _MIRROR_JOINTS.items() if l.startswith("left_")]
  asym = np.mean([np.abs(mean_q[:, i] - sgn * mean_q[:, j]) for i, j, sgn in pairs], axis=0)
  seg = {}
  for k in seg_bounds:
    mq = (seg_q[k] / seg_c[k].clamp(min=1)[:, None]).cpu().numpy()
    has = seg_c[k].cpu().numpy() > 0
    per = np.array([np.abs(mq[:, i] - sgn * mq[:, j]) for i, j, sgn in pairs])  # [pairs, n]
    a_k = per.mean(0)[has]
    worst = int(np.argmax(np.median(per[:, has], axis=1))) if has.any() else 0
    seg[k] = {"median": float(np.median(a_k)) if has.any() else float("nan"),
              "p90": float(np.percentile(a_k, 90)) if has.any() else float("nan"),
              "worst_pair": names[pairs[worst][0]][5:],
              "worst_median": float(np.median(per[worst, has])) if has.any() else float("nan")}
  succ = st_np == 6
  tu = t_up.cpu().numpy()
  q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
  return {
    "stage_dist": {STAGES[k]: float((st_np == k).mean()) for k in range(len(STAGES))},
    "stage_mean": float(st_np.mean()),
    "success": float(succ.mean()),
    "getup_time_median": q(tu[succ], 50), "getup_time_p90": q(tu[succ], 90),
    "flip_rate": float(flipped.float().mean()),
    "asym_median": float(np.median(asym)), "asym_p90": q(asym, 90), "asym_segments": seg,
    "torque_max_median": float(tau_max.median()), "torque_max": float(tau_max.max()),
    "qvel_max_median": float(vel_max.median()), "qvel_max": float(vel_max.max()),
    "walk_start": {k: float(v.float().mean()) for k, v in walk_at.items()},
    "pitch_end": {"median": float(pitch_end.median()), "p10": float(pitch_end.quantile(0.1)),
                  "p90": float(pitch_end.quantile(0.9))},
    "walk_start_given_success": float((walk_at["after_getup"] & torch.tensor(succ, device=dev)).float().sum()
                                      / max(1, succ.sum())),
  }


def fmt(tag: str, r: dict) -> str:
  dist = " ".join(f"{k} {v * 100:.0f}%" for k, v in r["stage_dist"].items())
  ws = " ".join(f"{k} {v * 100:.0f}%" for k, v in r["walk_start"].items())
  return (f"[{tag}] last stage passed: {dist} | success {r['success'] * 100:.1f}%, get-up time "
          f"{r['getup_time_median']:.2f} s (p90 {r['getup_time_p90']:.2f}), flipped {r['flip_rate'] * 100:.1f}% | "
          f"asym median {r['asym_median']:.3f} (p90 {r['asym_p90']:.3f}) rad, per segment " +
          ", ".join(f"{k} {v['median']:.3f} (worst {v['worst_pair']} {v['worst_median']:.3f})"
                    for k, v in r.get("asym_segments", {}).items()) + " | torque max median "
          f"{r['torque_max_median']:.2f} / max {r['torque_max']:.2f} Nm | qvel max median {r['qvel_max_median']:.1f} / "
          f"max {r['qvel_max']:.1f} rad/s | walk.onnx start ok: {ws} | pitch at end {r['pitch_end']['median']:+.3f} "
          f"[{r['pitch_end']['p10']:+.3f}, {r['pitch_end']['p90']:+.3f}] rad")


def alive_pid(pid: int | None) -> bool:
  if pid is None:
    return False
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False


def main(a: Args) -> None:
  cfg = load_env_cfg(a.task_id, play=True)
  cfg.scene.num_envs = a.num_envs
  agent_cfg = load_rl_cfg(a.task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if a.checkpoint_file:
    r = evaluate(env, vec_env, agent_cfg, a.checkpoint_file, a.seed, a.sequential)
    print(fmt(Path(a.checkpoint_file).name, r), flush=True)
    print("RESULT " + json.dumps(r), flush=True)
    return
  out = Path(a.run_dir) / "getup_monitor.jsonl"
  best_stage, best_succ, stall = -1.0, -1.0, 0
  for it in [*range(a.first, a.last, a.every), a.last]:
    ckpt = Path(a.run_dir) / f"model_{it}.pt"
    while not ckpt.exists():
      if not alive_pid(a.train_pid):
        print(f"training process ended before {ckpt.name}", flush=True)
        return
      time.sleep(20)
    time.sleep(5)
    r = evaluate(env, vec_env, agent_cfg, str(ckpt), a.seed, a.sequential)
    with out.open("a") as f:
      f.write(json.dumps({"iteration": it, **r}) + "\n")
    print(fmt(f"it {it}", r), flush=True)
    why = []
    if r["asym_median"] > a.stop_asym:
      why.append(f"asymmetry median {r['asym_median']:.3f} rad > {a.stop_asym}")
    if r["asym_segments"]["push"]["median"] > a.stop_push_asym:
      why.append(f"push-segment asymmetry median {r['asym_segments']['push']['median']:.3f} rad > {a.stop_push_asym}")
    if (r["stage_mean"] >= best_stage + MIN_GAIN_STAGE or r["success"] >= best_succ + MIN_GAIN_SUCCESS
        or r["success"] >= a.stall_below_success):
      stall = 0
    else:
      stall += 1
    best_stage, best_succ = max(best_stage, r["stage_mean"]), max(best_succ, r["success"])
    if stall >= STALL_EVALS:
      why.append(f"no progress for {stall} evaluations (mean stage {r['stage_mean']:.2f}, best "
                 f"{best_stage:.2f}; success {r['success'] * 100:.1f}%, best {best_succ * 100:.1f}%)")
    if r["success"] < a.min_success:
      why.append(f"success {r['success'] * 100:.1f}% < {a.min_success * 100:.0f}%")
    if why:
      print(f"STOP at it {it}: {'; '.join(why)}", flush=True)
      if alive_pid(a.train_pid):
        os.kill(a.train_pid, signal.SIGINT)
        print(f"sent SIGINT to training pid {a.train_pid}", flush=True)
      return
  print("monitor finished", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Args))
