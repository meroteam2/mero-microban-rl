"""Per-environment gait checks of a Microban-Crawl-Arms run (or one checkpoint).

Rolls a checkpoint out (play config, --seconds, the first SKIP_S dropped) and
prints per-environment statistics -- nothing is averaged across environments
before taking a magnitude:

- posture asymmetry: mean over joint pairs of |mean_t q_L - sign * mean_t q_R|
  (all joints, and arm joints only)
- hand / elbow / foot ground-contact ratio per side (any arm / leg part too)
- arm slip: horizontal speed of the forearm contact point while it touches
- arm and leg cycle: dominant frequency of the left arm and left leg contact
  signals; diagonal timing: P(diagonal leg down | arm down) vs. P(same-side leg
  down | arm down), and the phase of each arm's contact signal relative to the
  diagonal leg at the leg frequency (0 deg = in phase)
- arm share of the forward (+x) ground impulse
- forward speed, peak joint speed

Watch mode (run dir): waits for model_<k>.pt every --every iterations from
--first, and stops --train-pid when the asymmetry median exceeds STOP_ASYM rad
or the mean forward speed drops below STOP_SPEED m/s.

  uv run python scripts/monitor_arms.py logs/rsl_rl/microban_crawl_arms/<run> --train-pid <pid>
  uv run python scripts/monitor_arms.py --checkpoint-file <model.pt>
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
from microban_rl.crawl_arms_env_cfg import ARM_JOINTS, LIMB_BODIES, limb_contacts
from microban_rl.crawl_cpg_env_cfg import ARM_CONTACT_SENSOR, get_spec_with_arm_geom_names
from microban_rl.crawl_env_cfg import GROUND_FORCE_SENSOR, contact_slip_speed
from microban_rl.prone import get_microban_prone_robot_cfg
from microban_rl.symmetry import _MIRROR_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

STOP_ASYM = 0.10  # rad
STOP_SPEED = 0.10  # m/s
SKIP_S = 4.0


@dataclass(frozen=True)
class Args:
  run_dir: tyro.conf.Positional[str] = ""
  checkpoint_file: str = ""
  task_id: str = "Microban-Crawl-Arms"
  train_pid: int | None = None
  first: int = 2000
  every: int = 500
  last: int = 3498
  num_envs: int = 128
  seconds: float = 16.0


def _dominant(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
  """Per-env dominant frequency [N] and complex spectrum [F, N] of x [T, N]."""
  x = x - x.mean(0)
  spec = np.fft.rfft(x, axis=0)
  freqs = np.fft.rfftfreq(x.shape[0], dt)
  k = np.abs(spec[1:]).argmax(0) + 1
  return freqs[k], spec


def evaluate(env, vec_env, agent_cfg, ckpt: str, seconds: float) -> dict:
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=env.device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)
  with torch.inference_mode():  # sensor buffers were created in inference mode
    env.reset()
    obs = vec_env.get_observations()
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  ground, arm = env.scene["ground_contact"], env.scene["arm_contact"]
  force = env.scene[GROUND_FORCE_SENSOR.name]
  gnames = list(ground.primary_names)
  fnames = list(force.primary_names)
  arm_g = [gnames.index(b) for b in ("radius_2", "radius")]
  arm_f = [fnames.index(b) for b in ("radius_2", "radius")]
  skip = int(SKIP_S / env.step_dt)
  start = None
  q_log, v_log, limb_log, part_log, foot_log, slip_log, f_log = [], [], [], [], [], [], []
  with torch.inference_mode():
    for t in range(int(seconds / env.step_dt)):
      obs, _, _, _ = vec_env.step(policy(obs))
      if t < skip:
        continue
      if start is None:
        start = robot.data.root_link_pos_w[:, 0].clone()
      q_log.append((robot.data.joint_pos - robot.data.default_joint_pos).cpu())
      v_log.append(robot.data.joint_vel.abs().cpu())
      c = limb_contacts(env)
      limb_log.append(torch.stack([c[k] for k in LIMB_BODIES], -1).cpu())
      part_log.append((arm.data.found > 0).cpu())
      foot_log.append((ground.data.found[:, [gnames.index("foot_2"), gnames.index("foot")]] > 0).cpu())
      s, inc = contact_slip_speed(env)
      slip_log.append(torch.stack([s[:, arm_g], inc[:, arm_g].float()], -1).cpu())
      f_log.append(force.data.force_history.clone().cpu())  # [B, P, H, 3]
    dur = len(q_log) * env.step_dt
    speed = ((robot.data.root_link_pos_w[:, 0] - start) / dur).cpu().numpy()

  dt = env.step_dt
  q = torch.stack(q_log).numpy()
  v = torch.stack(v_log).numpy()
  limb = torch.stack(limb_log).numpy().astype(float)  # [T, N, 4] arm_L arm_R leg_L leg_R
  part = torch.stack(part_log).numpy().mean(0)
  foot = torch.stack(foot_log).numpy().mean(0)
  slip = torch.stack(slip_log).numpy()  # [T, N, 2, 2]

  m = q.mean(0)
  pairs = [(n, p, s) for n, (p, s) in _MIRROR_JOINTS.items() if n.startswith("left_")]

  def asym_of(ps):
    return np.mean([np.abs(m[:, names.index(n)] - s * m[:, names.index(p)]) for n, p, s in ps], 0)

  asym = asym_of(pairs)
  asym_arm = asym_of([x for x in pairs if x[0][5:] in ARM_JOINTS])

  ai = {n: i for i, n in enumerate(arm.primary_names)}
  aL, aR, lL, lR = (limb[..., i] for i in range(4))
  ratio = limb.mean(0)  # [N, 4]

  # Arm slip speed while in contact, per env (NaN if that arm never touches).
  sp, inc = slip[..., 0], slip[..., 1]
  arm_slip = (sp * inc).sum(0) / np.maximum(inc.sum(0), 1)  # [N, 2]
  arm_slip[inc.sum(0) < 5] = np.nan

  # Cycles and diagonal timing.
  f_arm, s_aL = _dominant(aL, dt)
  f_leg, s_lL = _dominant(lL, dt)
  _, s_aR = _dominant(aR, dt)
  _, s_lR = _dominant(lR, dt)
  kleg = np.abs(s_lL[1:]).argmax(0) + 1
  cols = np.arange(aL.shape[1])

  def phase(a, b):  # phase of a relative to b at the leg frequency, deg
    return np.degrees(np.angle(a[kleg, cols] * np.conj(b[kleg, cols])))

  ph_L = phase(s_aL, s_lR)  # left arm vs right leg
  ph_R = phase(s_aR, s_lL)
  arm_active = (ratio[:, 0] > 0.05) & (ratio[:, 1] > 0.05)

  def cond(a, b):
    return (a * b).sum(0) / np.maximum(a.sum(0), 1)

  p_diag = 0.5 * (cond(aL, lR) + cond(aR, lL))
  p_ipsi = 0.5 * (cond(aL, lL) + cond(aR, lR))

  # Forward (+x) ground impulse share of the arms.
  f = torch.stack(f_log).numpy()
  f = f * np.sign(np.nanmean(f[..., 2]) or 1.0)
  push = np.clip(f[..., 0], 0, None)  # [T, N, P, H]
  arm_push_share = push[:, :, arm_f].sum() / max(push.sum(), 1e-9)
  fx = f[..., 0].sum(-1)  # [T, N, P]
  arm_net_fx = fx[:, :, arm_f].sum(-1).mean()
  leg_net_fx = fx.sum(-1).mean() - arm_net_fx

  def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

  def frac(x):
    return float(np.mean(x))

  return {
    "speed_mean": float(speed.mean()), "speed_p10": float(np.percentile(speed, 10)),
    "asym_median": float(np.median(asym)), "asym_p90": float(np.percentile(asym, 90)),
    "asym_frac_over_0.3": frac(asym > 0.3),
    "arm_asym_median": float(np.median(asym_arm)), "arm_asym_p90": float(np.percentile(asym_arm, 90)),
    "arm_L": float(ratio[:, 0].mean()), "arm_R": float(ratio[:, 1].mean()),
    "leg_L": float(ratio[:, 2].mean()), "leg_R": float(ratio[:, 3].mean()),
    "arm_LR_diff_median": float(np.median(np.abs(ratio[:, 0] - ratio[:, 1]))),
    "hand_L": float(part[:, ai["left_hand_collision"]].mean()),
    "hand_R": float(part[:, ai["right_hand_collision"]].mean()),
    "elbow_L": float(part[:, ai["left_elbow_collision"]].mean()),
    "elbow_R": float(part[:, ai["right_elbow_collision"]].mean()),
    "foot_L": float(foot[:, 0].mean()), "foot_R": float(foot[:, 1].mean()),
    "arm_slip_L_median": med(arm_slip[:, 0]), "arm_slip_R_median": med(arm_slip[:, 1]),
    "arm_active_frac": frac(arm_active),
    "f_arm_median": med(np.where(arm_active, f_arm, np.nan)), "f_leg_median": med(f_leg),
    "same_freq_frac": frac(arm_active & (np.abs(f_arm - f_leg) <= 0.1)),
    "p_diag_given_arm_median": med(np.where(arm_active, p_diag, np.nan)),
    "p_ipsi_given_arm_median": med(np.where(arm_active, p_ipsi, np.nan)),
    "phase_armL_legR_median_abs": med(np.where(arm_active, np.abs(ph_L), np.nan)),
    "phase_armR_legL_median_abs": med(np.where(arm_active, np.abs(ph_R), np.nan)),
    "arm_push_share": float(arm_push_share),
    "arm_net_fx_W": float(arm_net_fx), "rest_net_fx_W": float(leg_net_fx),
    "qvel_max": float(v.max()), "qvel_p99": float(np.percentile(v, 99)),
  }


def fmt(tag, r: dict) -> str:
  p = lambda k: f"{r[k] * 100:.0f}"  # noqa: E731
  return (
    f"[{tag}] speed {r['speed_mean']:.3f} (p10 {r['speed_p10']:.3f}) m/s | asym med "
    f"{r['asym_median']:.3f} p90 {r['asym_p90']:.3f} (arm med {r['arm_asym_median']:.3f}) rad | "
    f"arm L/R {p('arm_L')}/{p('arm_R')}% (|L-R| med {r['arm_LR_diff_median']:.2f}) hand "
    f"{p('hand_L')}/{p('hand_R')}% elbow {p('elbow_L')}/{p('elbow_R')}% foot "
    f"{p('foot_L')}/{p('foot_R')}% leg {p('leg_L')}/{p('leg_R')}% | arm slip L/R "
    f"{r['arm_slip_L_median']:.3f}/{r['arm_slip_R_median']:.3f} m/s | f arm/leg "
    f"{r['f_arm_median']:.2f}/{r['f_leg_median']:.2f} Hz same {p('same_freq_frac')}% of envs | "
    f"P(diag|arm) {r['p_diag_given_arm_median']:.2f} P(ipsi|arm) {r['p_ipsi_given_arm_median']:.2f} "
    f"phase armL-legR {r['phase_armL_legR_median_abs']:.0f} armR-legL "
    f"{r['phase_armR_legL_median_abs']:.0f} deg | arm +x push share {p('arm_push_share')}% | "
    f"qvel max {r['qvel_max']:.1f} p99 {r['qvel_p99']:.1f}"
  )


def alive(pid: int | None) -> bool:
  if pid is None:
    return True
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False


def main(run_dir: str, checkpoint_file: str, task_id: str, train_pid: int | None, first: int,
         every: int, last: int, num_envs: int, seconds: float) -> None:
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  # Hand / elbow sensor (sensing only).
  cfg.scene.entities["robot"] = get_microban_prone_robot_cfg(spec_fn=get_spec_with_arm_geom_names)
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (ARM_CONTACT_SENSOR,)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if checkpoint_file:
    r = evaluate(env, vec_env, agent_cfg, checkpoint_file, seconds)
    print(fmt(Path(checkpoint_file).name, r), flush=True)
    print("RESULT " + json.dumps(r), flush=True)
    return

  out = Path(run_dir) / "arms_monitor.jsonl"
  for it in [*range(first, last, every), last]:
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
    print(fmt(f"it {it}", r), flush=True)
    why = []
    if r["asym_median"] > STOP_ASYM:
      why.append(f"asymmetry median {r['asym_median']:.3f} rad > {STOP_ASYM}")
    if r["speed_mean"] < STOP_SPEED:
      why.append(f"speed {r['speed_mean']:.3f} m/s < {STOP_SPEED}")
    if why:
      print(f"STOP at it {it}: {'; '.join(why)}", flush=True)
      if train_pid is not None and alive(train_pid):
        os.kill(train_pid, signal.SIGINT)
        print(f"sent SIGINT to training pid {train_pid}", flush=True)
      return
  print("monitor finished", flush=True)


if __name__ == "__main__":
  main(**asdict(tyro.cli(Args)))
