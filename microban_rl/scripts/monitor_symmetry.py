"""Per-environment gait checks of a training run, every N iterations.

Waits for model_<k>.pt (k = every, 2 * every, ..., plus the final one), rolls
it out (play config, 10 s) and prints per-environment statistics -- nothing is
averaged across environments before taking a magnitude:

- posture asymmetry: mean over joint pairs of |mean_t q_L - sign * mean_t q_R|
- half-cycle mirror error: RMS of q(t) - mirror(q(t - T/2)), T = CPG period
- head angle |mean_t q_head| and which way it is turned
- hand / foot / elbow ground-contact ratios per side
- forward speed

Stops the training process (--train-pid) when asymmetry starts growing again:
posture asymmetry median rises by more than STOP_RISE rad over the lowest
value seen so far, or exceeds STOP_ABS rad from iteration 1000 on.

  uv run python scripts/monitor_symmetry.py logs/rsl_rl/microban_crawl_cpg/<run> \
      --train-pid <pid>
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.symmetry import _MIRROR_JOINTS, mirror_joint_map
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

STOP_RISE = 0.10  # rad
STOP_ABS = 0.30  # rad


@dataclass(frozen=True)
class Args:
  run_dir: tyro.conf.Positional[str]
  task_id: str = "Microban-Crawl-CPG"
  train_pid: int | None = None
  every: int = 500
  last: int = 2999
  num_envs: int = 128
  seconds: float = 10.0


def evaluate(env, vec_env, agent_cfg, ckpt: str, seconds: float) -> dict:
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=env.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)
  env.reset()
  obs = vec_env.get_observations()
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  ground, arm = env.scene["ground_contact"], env.scene["arm_contact"]
  cpg = env.action_manager.get_term("cpg")
  start = robot.data.root_link_pos_w[:, 0].clone()
  q_log, f_log, g_log, a_log = [], [], [], []
  steps = int(seconds / env.step_dt)
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, _, _ = vec_env.step(policy(obs))
      q_log.append((robot.data.joint_pos - robot.data.default_joint_pos).cpu())
      f_log.append(cpg.frequency.cpu())
      g_log.append((ground.data.found > 0).cpu())
      a_log.append((arm.data.found > 0).cpu())
    speed = ((robot.data.root_link_pos_w[:, 0] - start) / seconds).cpu().numpy()

  half = steps // 2  # skip the start-up transient
  q = torch.stack(q_log).numpy()[half:]  # [T, N, J]
  freq = torch.stack(f_log).numpy()[half:]
  ground_c = torch.stack(g_log).numpy()[half:].mean(0)  # [N, P]
  arm_c = torch.stack(a_log).numpy()[half:].mean(0)

  m = q.mean(0)  # [N, J] per-env mean offsets
  pairs = [(n, p, s) for n, (p, s) in _MIRROR_JOINTS.items() if n.startswith("left_")]
  asym = np.mean([np.abs(m[:, names.index(n)] - s * m[:, names.index(p)])
                  for n, p, s in pairs], axis=0)  # [N]
  head = m[:, names.index("head")]

  idx, sign = (t.cpu().numpy() for t in mirror_joint_map(names, "cpu"))
  lag = np.clip(np.rint(0.5 / (freq.mean(0) * env.step_dt)).astype(int), 1, q.shape[0] // 2)
  hc = np.array([np.sqrt(np.mean((q[l:, e] - q[:-l, e][:, idx] * sign) ** 2))
                 for e, l in enumerate(lag)])

  gi = {n: i for i, n in enumerate(ground.primary_names)}
  ai = {n: i for i, n in enumerate(arm.primary_names)}
  contact = {
    "hand_L": arm_c[:, ai["left_hand_collision"]], "hand_R": arm_c[:, ai["right_hand_collision"]],
    "elbow_L": arm_c[:, ai["left_elbow_collision"]], "elbow_R": arm_c[:, ai["right_elbow_collision"]],
    "foot_L": ground_c[:, gi["foot_2"]], "foot_R": ground_c[:, gi["foot"]],
  }

  def imbalance(l, r):
    return np.abs(l - r) / np.maximum(l + r, 1e-3)

  return {
    "speed_mean": float(speed.mean()), "speed_p10": float(np.percentile(speed, 10)),
    "asym_median": float(np.median(asym)), "asym_p90": float(np.percentile(asym, 90)),
    "asym_frac_over_0.3": float(np.mean(asym > 0.3)),
    "half_cycle_rms_median": float(np.median(hc)), "half_cycle_rms_p90": float(np.percentile(hc, 90)),
    "head_abs_median": float(np.median(np.abs(head))), "head_abs_max": float(np.abs(head).max()),
    "head_turned_plus_frac": float(np.mean(head > 0.05)),
    "head_turned_minus_frac": float(np.mean(head < -0.05)),
    "cpg_freq_mean": float(freq.mean()),
    **{f"{k}_mean": float(v.mean()) for k, v in contact.items()},
    "hand_LR_imbalance_median": float(np.median(imbalance(contact["hand_L"], contact["hand_R"]))),
    "foot_LR_imbalance_median": float(np.median(imbalance(contact["foot_L"], contact["foot_R"]))),
  }


def fmt(it: int, r: dict) -> str:
  return (f"[it {it}] speed {r['speed_mean']:.3f} (p10 {r['speed_p10']:.3f}) m/s | "
          f"asym med {r['asym_median']:.3f} p90 {r['asym_p90']:.3f} rad "
          f"(>0.3: {r['asym_frac_over_0.3'] * 100:.0f}%) | half-cycle RMS med "
          f"{r['half_cycle_rms_median']:.3f} | head |q| med {r['head_abs_median']:.2f} max "
          f"{r['head_abs_max']:.2f} (+{r['head_turned_plus_frac'] * 100:.0f}%/"
          f"-{r['head_turned_minus_frac'] * 100:.0f}%) | hand L/R {r['hand_L_mean'] * 100:.0f}/"
          f"{r['hand_R_mean'] * 100:.0f}% foot L/R {r['foot_L_mean'] * 100:.0f}/"
          f"{r['foot_R_mean'] * 100:.0f}% elbow L/R {r['elbow_L_mean'] * 100:.0f}/"
          f"{r['elbow_R_mean'] * 100:.0f}% | imbalance hand {r['hand_LR_imbalance_median']:.2f} "
          f"foot {r['foot_LR_imbalance_median']:.2f} | f {r['cpg_freq_mean']:.2f} Hz")


def alive(pid: int | None) -> bool:
  if pid is None:
    return True
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False


def main(run_dir: str, task_id: str, train_pid: int | None, every: int, last: int,
         num_envs: int, seconds: float) -> None:
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  out = Path(run_dir) / "symmetry_monitor.jsonl"
  best = None
  for it in [*range(every, last + 1, every), last]:
    ckpt = Path(run_dir) / f"model_{it}.pt"
    while not ckpt.exists():
      if not alive(train_pid):
        print(f"training process ended before {ckpt.name}", flush=True)
        return
      time.sleep(20)
    time.sleep(5)  # let the save finish
    r = evaluate(env, vec_env, agent_cfg, str(ckpt), seconds)
    with out.open("a") as f:
      f.write(json.dumps({"iteration": it, **r}) + "\n")
    print(fmt(it, r), flush=True)
    a = r["asym_median"]
    rising = best is not None and a - best > STOP_RISE
    too_big = it >= 1000 and a > STOP_ABS
    if rising or too_big:
      why = (f"asymmetry rose {a - best:+.3f} rad over the best {best:.3f}" if rising
             else f"asymmetry {a:.3f} rad > {STOP_ABS}")
      print(f"STOP at it {it}: {why}", flush=True)
      if train_pid is not None and alive(train_pid):
        os.kill(train_pid, signal.SIGINT)
        print(f"sent SIGINT to training pid {train_pid}", flush=True)
      return
    best = a if best is None else min(best, a)
  print("monitor finished", flush=True)


if __name__ == "__main__":
  main(**asdict(tyro.cli(Args)))
