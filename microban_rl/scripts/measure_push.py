"""Hand-push symmetry of a Microban-GetUp checkpoint, per environment.

Rolls every env out from prone (Microban-GetUp-4 play config, which has the
forearm force sensor; the observations are those of all GetUp tasks) for one
full episode and measures in the push window (pike_1 -> squat_deep):

- max |trunk lateral roll| (asin of the world z of the trunk's left axis) and
  the signed roll at that moment (+ = left side up = leaning on the right arm)
- max |trunk yaw change| since pike_1 (signed at the maximum)
- hand lift-off: last time each forearm touches the floor in the window;
  left - right difference; share of the window with exactly one hand down
- left arm's share of the two forearms' normal impulse in the window
- twist direction per env: side the trunk leans to (sign of the roll at its
  maximum) and the arm carrying more load
- landing impact (tasks with the foot_force sensor, Microban-GetUp-5 on): peak
  per-substep normal force of one foot and of both feet together, from pike_1
  to the end, in body weights (the robot's weight, ~8 N)
- also: push-window left/right difference of hip_yaw (segment means) and the
  share of time at the torque limit per joint (whole episode)

Envs that flip onto their back are counted up to the flip.

  uv run python scripts/measure_push.py <model.pt> [<model.pt> ...]
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict

import numpy as np
import torch

import os

import microban_rl  # noqa: F401
from microban_rl.crawl_env_cfg import robot_weight
from microban_rl.getup4_env_cfg import push_metrics
from microban_rl.getup_env_cfg import HOLD_S, in_window, ref_time, reference
from microban_rl.robot import XL330_EFFORT_LIMIT
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

N = 256


def measure(env, ve, ag, ckpt: str) -> dict:
  torch.manual_seed(1)
  r = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=env.device)
  pol = r.get_inference_policy(device=env.device)
  robot = env.scene["robot"]
  names = [n.split("/")[-1] for n in robot.joint_names]
  dev = env.device
  iL, iR = names.index("left_hip_yaw"), names.index("right_hip_yaw")
  with torch.inference_mode():
    env.reset()
    obs = ve.get_observations()
    ref = reference(env)
    steps = int(round((ref.t_end + HOLD_S + 0.5) / env.step_dt))
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    lat_max = torch.zeros(N, device=dev); lat_sgn = torch.zeros(N, device=dev)
    yaw_max = torch.zeros(N, device=dev); yaw_sgn = torch.zeros(N, device=dev)
    last_down = torch.full((N, 2), float("nan"), device=dev)
    one, win_n = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    imp = torch.zeros(N, 2, device=dev)
    hy = torch.zeros(N, 2, device=dev)
    sat = torch.zeros(len(names), device=dev); alive_n = torch.zeros((), device=dev)
    has_ff = "foot_force" in env.scene.sensors
    foot_pk = torch.zeros(N, device=dev); feet_pk = torch.zeros(N, device=dev)
    t_pk = torch.full((N,), float("nan"), device=dev)
    for _ in range(steps):
      obs, _, d, _ = ve.step(pol(obs))
      alive &= ~d.bool()
      m = push_metrics(env)
      w = in_window(env) & alive
      wf = w.float()
      t = ref_time(env)
      a = m["lat"].abs()
      up = w & (a > lat_max)
      lat_max = torch.where(up, a, lat_max); lat_sgn = torch.where(up, torch.sign(m["lat"]), lat_sgn)
      b = m["dyaw"].abs()
      up = w & (b > yaw_max)
      yaw_max = torch.where(up, b, yaw_max); yaw_sgn = torch.where(up, torch.sign(m["dyaw"]), yaw_sgn)
      last_down = torch.where(w[:, None] & m["hands"], t[:, None], last_down)
      one += ((m["hands"].sum(1) == 1).float() * wf)
      win_n += wf
      imp += m["force"] * wf[:, None] * env.step_dt
      hy += torch.stack([robot.data.joint_pos[:, iL], robot.data.joint_pos[:, iR]], 1) * wf[:, None]
      at = (robot.data.actuator_force.abs() >= 0.98 * XL330_EFFORT_LIMIT).float() * alive.float()[:, None]
      sat += at.sum(0); alive_n += alive.float().sum()
      if has_ff:
        fz = env.scene["foot_force"].data.force_history[..., 2].abs()  # [B, 2, H]
        late = alive & (t >= ref.kf_t["pike_1"])
        one_ = fz.amax((1, 2)); both = fz.sum(1).amax(-1)
        up = late & (both > feet_pk)
        t_pk = torch.where(up, t, t_pk)
        foot_pk = torch.where(late, torch.maximum(foot_pk, one_), foot_pk)
        feet_pk = torch.where(late, torch.maximum(feet_pk, both), feet_pk)
  c = lambda x: x.cpu().numpy()  # noqa: E731
  has = c(win_n) > 0
  lat_deg, yaw_deg = np.degrees(c(lat_max))[has], np.degrees(c(yaw_max))[has]
  ld = c(last_down)[has]
  diff = ld[:, 0] - ld[:, 1]
  one_r = (c(one) / np.maximum(c(win_n), 1))[has]
  share_l = (c(imp)[:, 0] / np.maximum(c(imp).sum(1), 1e-9))[has]
  hym = c(hy)[has] / c(win_n)[has, None]
  hip_yaw_diff = np.abs(hym[:, 0] - (-1.0) * hym[:, 1])  # hip_yaw mirrors with sign -1
  lean = c(lat_sgn)[has]
  sat_share = c(sat) / max(float(alive_n), 1)
  top = sorted(((float(s), n) for s, n in zip(sat_share, names)), reverse=True)[:4]
  med = lambda x: float(np.nanmedian(x)) if len(x) else float("nan")  # noqa: E731
  p90 = lambda x: float(np.nanpercentile(x, 90)) if len(x) else float("nan")  # noqa: E731
  return {
    "envs_in_window": int(has.sum()),
    "trunk_roll_max_deg": {"median": med(lat_deg), "p90": p90(lat_deg)},
    "trunk_yaw_change_max_deg": {"median": med(yaw_deg), "p90": p90(yaw_deg)},
    "hand_liftoff_L_minus_R_s": {"median": med(diff), "abs_median": med(np.abs(diff))},
    "one_hand_share": {"median": med(one_r), "p90": p90(one_r)},
    "left_arm_load_share": {"median": med(share_l), "p10": float(np.nanpercentile(share_l, 10)),
                            "p90": p90(share_l)},
    "lean_to": {"right_arm (left side up)": float((lean > 0).mean()), "left_arm (right side up)": float((lean < 0).mean())},
    "heavier_arm": {"left": float((share_l > 0.55).mean()), "right": float((share_l < 0.45).mean()),
                    "balanced": float(((share_l >= 0.45) & (share_l <= 0.55)).mean())},
    "by_heavier_arm": {
      side: {"share": float(sel.mean()), "liftoff_L_minus_R_s": med(diff[sel]), "roll_max_deg": med(lat_deg[sel]),
             "roll_sign_left_up": float((lean[sel] > 0).mean()) if sel.any() else float("nan"),
             "yaw_change_signed_deg": med(np.degrees(c(yaw_max)[has][sel] * c(yaw_sgn)[has][sel]))}
      for side, sel in (("left", share_l > 0.5), ("right", share_l <= 0.5))},
    "push_hip_yaw_diff_median": med(hip_yaw_diff),
    "torque_limit_share_top": top,
    **({"landing_peak_bw": {
      "one_foot": {"median": med(c(foot_pk) / robot_weight(env)), "p90": p90(c(foot_pk) / robot_weight(env)),
                   "max": float(c(foot_pk).max() / robot_weight(env))},
      "both_feet": {"median": med(c(feet_pk) / robot_weight(env)), "p90": p90(c(feet_pk) / robot_weight(env)),
                    "max": float(c(feet_pk).max() / robot_weight(env))},
      "time_of_peak_s": med(c(t_pk))}} if has_ff else {}),
  }


def fmt(tag: str, r: dict) -> str:
  return (f"[{tag}] push window ({r['envs_in_window']} envs): trunk roll max {r['trunk_roll_max_deg']['median']:.1f} deg "
          f"(p90 {r['trunk_roll_max_deg']['p90']:.1f}), yaw change max {r['trunk_yaw_change_max_deg']['median']:.1f} deg "
          f"(p90 {r['trunk_yaw_change_max_deg']['p90']:.1f}) | hand lift-off L-R {r['hand_liftoff_L_minus_R_s']['median']:+.2f} s, "
          f"one hand down {r['one_hand_share']['median'] * 100:.0f}% of the window | left arm load share "
          f"{r['left_arm_load_share']['median'] * 100:.0f}% (p10 {r['left_arm_load_share']['p10'] * 100:.0f}, p90 "
          f"{r['left_arm_load_share']['p90'] * 100:.0f}) | leans onto: right arm {r['lean_to']['right_arm (left side up)'] * 100:.0f}% / "
          f"left arm {r['lean_to']['left_arm (right side up)'] * 100:.0f}%; heavier arm L {r['heavier_arm']['left'] * 100:.0f}% / "
          f"R {r['heavier_arm']['right'] * 100:.0f}% / even {r['heavier_arm']['balanced'] * 100:.0f}% | push hip_yaw diff "
          f"{r['push_hip_yaw_diff_median']:.3f} rad | at torque limit: "
          + (", ".join(f"{n} {s * 100:.1f}%" for s, n in r["torque_limit_share_top"] if s > 0.0005) or "none")
          + (f" | landing peak: one foot {r['landing_peak_bw']['one_foot']['median']:.1f} x BW (p90 "
             f"{r['landing_peak_bw']['one_foot']['p90']:.1f}, max {r['landing_peak_bw']['one_foot']['max']:.1f}), both feet "
             f"{r['landing_peak_bw']['both_feet']['median']:.1f} x BW (p90 {r['landing_peak_bw']['both_feet']['p90']:.1f}, max "
             f"{r['landing_peak_bw']['both_feet']['max']:.1f}) at t {r['landing_peak_bw']['time_of_peak_s']:.2f} s"
             if "landing_peak_bw" in r else "")
          + " | by heavier arm: " + "; ".join(
            f"{k} {v['share'] * 100:.0f}% (lift-off L-R {v['liftoff_L_minus_R_s']:+.2f} s, roll {v['roll_max_deg']:.0f} deg, "
            f"yaw {v['yaw_change_signed_deg']:+.0f} deg)" for k, v in r["by_heavier_arm"].items()))


def main() -> None:
  task = os.environ.get("TASK", "Microban-GetUp-5")
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = N
  ag = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  for ck in sys.argv[1:]:
    r = measure(env, ve, ag, ck)
    print(fmt(ck.split("/")[-1], r), flush=True)
    print("RESULT " + json.dumps({"checkpoint": ck, **r}), flush=True)


if __name__ == "__main__":
  main()
