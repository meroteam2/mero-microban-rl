"""Stage 4: get up from prone with a GetUp policy, hold, switch to walk.onnx and walk.

Per env (all start prone, prone.py randomization, random heading):
  1. get-up policy (--checkpoint, a Microban-GetUp* checkpoint) until the
     reference end + --getup-extra-s;
  2. hold the default pose (= stand_target joints) for --hold-s at robot.py's
     gain (kp 0.88), like microban before a move starts (--hold-mode pd: fixed
     targets; policy: the get-up policy keeps balancing around stand_target);
  3. switch the motor gain to walk.onnx's KP_RL (0.275 Nm/rad) and run walk.onnx
     (microban's sensor delays) at vx --vx for --walk-s; optional weak heading
     hold vtheta = -heading_gain * (yaw change since the switch), |vtheta| <= 0.3.
The torque limit (0.64 Nm) and the 10.3 rad/s target speed limit apply
throughout (the GetUp action term), joint ranges as in robot.py.

Reports per env: walk.onnx start conditions at the switch (stage 1: |roll| < 0.2,
pitch -0.1..+0.5, joint RMS < 0.12, soles down, nearly still), falls while
walking (trunk tilt > 60 deg or height < 0.10 m, as walk_onnx_check), forward
speed, lateral drift, heading change, steps, peak foot force; transition
success = start conditions met + no fall + forward speed > 0.05 m/s.

  uv run python scripts/getup_walk.py --checkpoint <model.pt> --num-envs 32
  uv run python scripts/getup_walk.py --checkpoint <model.pt> --viewer viser   # http://localhost:8080
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.crawl_env_cfg import robot_weight
from microban_rl.getup_env_cfg import ACTION_SCALE, getup_state, ref_time, reference
from microban_rl.robot import XL330_DAMPING, XL330_STIFFNESS
from microban_rl.walk_onnx import KP_RL_STIFFNESS, WalkOnnxPolicy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr import pd_gains
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply

import monitor_getup as M  # scripts/ is on sys.path when run as a script

FALL_GRAVITY_Z, FALL_HEIGHT = -0.5, 0.10


@dataclass(frozen=True)
class Args:
  checkpoint: str
  task: str = "Microban-GetUp-5"
  num_envs: int = 32
  vx: float = 0.15
  heading_gain: float = 1.0
  getup_extra_s: float = 1.0
  hold_s: float = 1.0
  hold_mode: str = "policy"
  """'pd': fixed default-pose targets (the gain alone holds it); 'policy': the get-up policy keeps
  balancing around stand_target (its trained post-reference hold)."""
  walk_s: float = 6.0
  seed: int = 3
  viewer: str = "none"  # "none" (measure) or "viser"


class GetUpThenWalk:
  """Callable policy: obs -> actions of the GetUp residual action term, phase per env."""

  def __init__(self, env: ManagerBasedRlEnv, getup_policy, a: Args):
    self.env, self.getup, self.a = env, getup_policy, a
    self.robot = env.scene["robot"]
    ref = reference(env)
    self.t_hold = ref.t_end + a.getup_extra_s
    self.t_walk = self.t_hold + a.hold_s
    self.t_stop = self.t_walk + a.walk_s
    term = env.action_manager.get_term("joint_pos")
    names = [n.split("/")[-1] for n in self.robot.joint_names]
    self.perm = torch.tensor([names.index(n.split("/")[-1]) for n in term.target_names], device=env.device)
    self.walk = WalkOnnxPolicy(self.robot, env.num_envs, env.device, delays=True)
    self.low_gain = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.walking = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.yaw0 = torch.zeros(env.num_envs, device=env.device)
    self.cmd = torch.zeros(env.num_envs, 3, device=env.device)

  def heading(self) -> torch.Tensor:
    ex = torch.tensor([1.0, 0, 0], device=self.env.device).expand(self.env.num_envs, 3)
    f = quat_apply(self.robot.data.root_link_quat_w, ex)  # belly / toe direction when standing
    return torch.atan2(f[:, 1], f[:, 0])

  def _set_gain(self, ids: torch.Tensor, kp: float) -> None:
    if len(ids):
      pd_gains(self.env, ids.int(), kp_range=(kp, kp), kd_range=(XL330_DAMPING, XL330_DAMPING), operation="abs")

  def phase(self) -> torch.Tensor:
    """0 get-up, 1 hold, 2 walk (by the reference time this step will end at)."""
    t = ref_time(self.env) + self.env.step_dt
    return (t >= self.t_hold).long() + (t >= self.t_walk).long()

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    ph = self.phase()
    # Episode restarted (e.g. by the viewer): back to the get-up gain.
    back = self.low_gain & (ph < 2)
    self._set_gain(back.nonzero().squeeze(-1), XL330_STIFFNESS)
    self.low_gain &= ~back
    self.walking &= ph == 2
    act = self.getup(obs)
    if self.a.hold_mode == "pd":
      act = torch.where((ph == 1)[:, None], torch.zeros_like(act), act)  # residual 0 = stand_target pose
    start = (ph == 2) & ~self.walking
    if start.any():
      ids = start.nonzero().squeeze(-1)
      self._set_gain(ids, KP_RL_STIFFNESS)
      self.low_gain |= start
      self.walk.reset(ids if self.walk._buf else None)
      self.yaw0[ids] = self.heading()[ids]
      self.walking |= start
    dyaw = torch.atan2(torch.sin(self.heading() - self.yaw0), torch.cos(self.heading() - self.yaw0))
    self.cmd[:, 0] = self.a.vx
    self.cmd[:, 2] = (-self.a.heading_gain * dyaw).clamp(-0.3, 0.3)
    w = self.walk.act(self.cmd)  # offsets from the default pose, robot joint order
    w_act = (w - (reference(self.env).q[-1] - reference(self.env).default)) / ACTION_SCALE
    act = torch.where(self.walking[:, None], w_act[:, self.perm], act)
    return act


def build(a: Args):
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.terminations = {}
  cfg.episode_length_s = 1e9
  # Makes the actuator gains per-env model fields, so they can be switched at runtime.
  cfg.events["gains"] = EventTermCfg(
    func=pd_gains, mode="startup",
    params={"kp_range": (XL330_STIFFNESS, XL330_STIFFNESS), "kd_range": (XL330_DAMPING, XL330_DAMPING),
            "operation": "abs"})
  ag = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
  runner = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=env.device)
  return env, ve, GetUpThenWalk(env, runner.get_inference_policy(device=env.device), a)


def measure(env, ve, ctl: GetUpThenWalk, a: Args) -> dict:
  n, dev = env.num_envs, env.device
  robot = env.scene["robot"]
  has_ff = "foot_force" in env.scene.sensors
  with torch.inference_mode():
    env.reset()
    obs = ve.get_observations()
    steps = int(round((ctl.t_stop - float(env._getup_t0[0])) / env.step_dt))
    start_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    start_pitch = torch.zeros(n, device=dev)
    fallen = torch.zeros(n, dtype=torch.bool, device=dev)
    fall_t = torch.full((n,), float("nan"), device=dev)
    p0 = h0 = None
    foot_prev = None
    touch = torch.zeros(n, device=dev)
    pk_walk = torch.zeros(n, device=dev)
    kp_seen = set()
    for s in range(steps):
      ph_next = ctl.phase()
      if p0 is None and bool((ph_next == 2).all()):
        st = getup_state(env)
        start_ok = M.walk_start_ok(env)
        start_pitch = st["pitch"].clone()
        p0 = robot.data.root_link_pos_w[:, :2].clone()
        h0 = ctl.heading().clone()
      obs, _, _, _ = ve.step(ctl(obs))
      kp_seen.add(round(float(env.sim.model.actuator_gainprm[0, 1, 0]), 3))
      if p0 is not None:
        g = robot.data.projected_gravity_b[:, 2]
        bad = (g > FALL_GRAVITY_Z) | (robot.data.root_link_pos_w[:, 2] < FALL_HEIGHT)
        fall_t = torch.where(bad & ~fallen, ref_time(env) - ctl.t_walk, fall_t)
        fallen |= bad
        st = getup_state(env)
        cs = env.scene["getup_contact"]
        pn = list(cs.primary_names)
        lf = cs.data.found[:, pn.index("foot_2")] > 0
        if foot_prev is not None:
          touch += (lf & ~foot_prev).float()
        foot_prev = lf
        if has_ff:
          pk_walk = torch.maximum(pk_walk, env.scene["foot_force"].data.force_history[..., 2].abs().amax((1, 2)))
    disp = robot.data.root_link_pos_w[:, :2] - p0
    hx, hy = torch.cos(h0), torch.sin(h0)
    fwd = disp[:, 0] * hx + disp[:, 1] * hy
    lat = -disp[:, 0] * hy + disp[:, 1] * hx
    dyaw = torch.atan2(torch.sin(ctl.heading() - h0), torch.cos(ctl.heading() - h0))
  speed = fwd / a.walk_s
  ok = start_ok & ~fallen & (speed > 0.05)
  c = lambda x: x.cpu().numpy()  # noqa: E731
  good = c(~fallen)
  med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
  r = {
    "num_envs": n, "checkpoint": a.checkpoint, "vx": a.vx, "heading_gain": a.heading_gain,
    "walk_start_ok_at_switch": float(start_ok.float().mean()),
    "pitch_at_switch": {"median": med(c(start_pitch)), "min": float(c(start_pitch).min()), "max": float(c(start_pitch).max())},
    "fell_while_walking": float(fallen.float().mean()),
    "fall_time_median_s": med(c(fall_t)[~good]) if (~good).any() else None,
    "transition_success": float(ok.float().mean()),
    "forward_speed_median": med(c(speed)[good]), "lateral_m_median": med(c(lat)[good]),
    "yaw_change_deg_median": med(np.degrees(c(dyaw)[good])),
    "left_foot_steps_median": med(c(touch)[good]),
    "gains_seen": sorted(kp_seen),
  }
  if has_ff:
    r["walk_foot_force_peak_bw"] = {"median": med(c(pk_walk)[good] / robot_weight(env)),
                                    "max": float(c(pk_walk).max() / robot_weight(env))}
  return r


def main(a: Args) -> None:
  torch.manual_seed(a.seed)
  env, ve, ctl = build(a)
  if a.viewer == "viser":
    from mjlab.viewer import ViserPlayViewer

    ViserPlayViewer(ve, ctl).run()
    return
  r = measure(env, ve, ctl, a)
  print(f"transition success {r['transition_success'] * 100:.1f}% ({a.num_envs} envs) | walk.onnx start ok at the "
        f"switch {r['walk_start_ok_at_switch'] * 100:.1f}% (pitch {r['pitch_at_switch']['median']:+.3f}) | fell while "
        f"walking {r['fell_while_walking'] * 100:.1f}% | forward {r['forward_speed_median']:.3f} m/s, lateral "
        f"{r['lateral_m_median']:+.2f} m, heading {r['yaw_change_deg_median']:+.1f} deg over {a.walk_s:.0f} s, "
        f"{r['left_foot_steps_median']:.0f} left steps | gains used {r['gains_seen']}"
        + (f" | walking foot force peak {r['walk_foot_force_peak_bw']['median']:.1f} x BW (max "
           f"{r['walk_foot_force_peak_bw']['max']:.1f})" if "walk_foot_force_peak_bw" in r else ""), flush=True)
  print("RESULT " + json.dumps(r), flush=True)


if __name__ == "__main__":
  main(tyro.cli(Args))
