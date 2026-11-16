"""Microban-GetUp-2: Microban-GetUp with the fixes for the one-sided hand push.

Run 1 (logs/rsl_rl/microban_getup/2026-11-16_19-49-21) reached squat_deep in
35% of the envs at iteration 500, then regressed by 1000; in the push window
(pike_1 -> squat_deep) the policy twisted to one side (hip_yaw up to 0.8 rad
left/right difference). Changes, otherwise identical to Microban-GetUp:

1. push_symmetry: per-step posture asymmetry (mean over joint pairs of
   |q_L - sign * q_R|, plus |head|) in the push window, weight -1.
2. Regularization ramp: REG_START (0.1) at the start of this run, rising to 1
   over REG2_RAMP_ITERATIONS (the run resumes from run 1's model_500).
3. Reference-state starts: keyframe share 25% -> 35% (replay 35% -> 25%),
   fold and squat_deep picked twice as often as pike_1 / pike_push / crouch.
4. (resume from model_500: see the README command.)
5. In the push window the residual of hip_yaw, hip_roll and ankle_roll is
   limited to +-PUSH_TWIST_LIMIT rad (other joints / times unchanged).
"""

from __future__ import annotations

import torch

from mjlab.managers.reward_manager import RewardTermCfg

from microban_rl.getup_env_cfg import (
  RSI_PRONE,
  REG_START,
  STEPS_PER_ITERATION,
  ReferenceResidualAction,
  ReferenceResidualActionCfg,
  in_window,
  microban_getup_env_cfg,
  microban_getup_ppo_runner_cfg,
  steps_since_start,
)
from microban_rl.symmetry import mirror_joint_map

REG2_RAMP_ITERATIONS = 1500
RSI2_KEYFRAME = 0.35
RSI2_KEYFRAME_WEIGHTS = {"pike_1": 1.0, "pike_push": 1.0, "fold": 2.0, "squat_deep": 2.0, "crouch": 1.0}
PUSH_TWIST_JOINTS = ("hip_yaw", "hip_roll", "ankle_roll")
PUSH_TWIST_LIMIT = 0.1  # rad


class PushLimitedResidualActionCfg(ReferenceResidualActionCfg):
  def build(self, env) -> PushLimitedResidualAction:
    return PushLimitedResidualAction(self, env)


class PushLimitedResidualAction(ReferenceResidualAction):
  """Residual of the twist joints clamped to +-PUSH_TWIST_LIMIT inside the push window."""

  def process_actions(self, actions: torch.Tensor) -> None:
    if not hasattr(self, "_twist_ids"):
      names = [n.split("/")[-1] for n in self.target_names]
      self._twist_ids = torch.tensor(
        [i for i, n in enumerate(names) if any(n.endswith(j) for j in PUSH_TWIST_JOINTS)], device=self.device)
    lim = PUSH_TWIST_LIMIT / self.cfg.scale
    # The window is evaluated at this step's target time (ref time + dt).
    self._env._getup_steps += 1
    win = in_window(self._env)
    self._env._getup_steps -= 1
    a = actions.clone()
    sub = a[:, self._twist_ids]
    a[:, self._twist_ids] = torch.where(win[:, None], sub.clamp(-lim, lim), sub)
    super().process_actions(a)


def push_symmetry(env) -> torch.Tensor:
  """Mean over mirror pairs of |q_L - sign q_R| (+ |head|), in the push window only."""
  robot = env.scene["robot"]
  cache = env.__dict__.get("_push_sym")
  if cache is None:
    names = [n.split("/")[-1] for n in robot.joint_names]
    idx, sign = mirror_joint_map(names, env.device)
    left = torch.tensor([i for i, n in enumerate(names) if n.startswith("left_")], device=env.device)
    head = names.index("head")
    cache = env._push_sym = (left, idx[left], sign[left], head)  # type: ignore[attr-defined]
  left, right, sign, head = cache
  q = robot.data.joint_pos
  d = (q[:, left] - sign * q[:, right]).abs()
  v = (d.sum(1) + q[:, head].abs()) / (len(left) + 1)
  return v * in_window(env).float()


def reg2_scale(env) -> float:
  t = steps_since_start(env) / (REG2_RAMP_ITERATIONS * STEPS_PER_ITERATION)
  return min(1.0, REG_START + (1.0 - REG_START) * t)


def microban_getup2_env_cfg(play: bool = False):
  cfg = microban_getup_env_cfg(play=play)
  old = cfg.actions["joint_pos"]
  cfg.actions = {"joint_pos": PushLimitedResidualActionCfg(
    entity_name=old.entity_name, actuator_names=old.actuator_names, scale=old.scale,
    use_default_offset=False, ramp_steps=0)}
  ev = cfg.events["reset_getup"].params
  if not play:
    ev["p_prone"], ev["p_replay"] = RSI_PRONE, 1.0 - RSI_PRONE - RSI2_KEYFRAME
    ev["keyframe_weights"] = dict(RSI2_KEYFRAME_WEIGHTS)
  for name in ("torques", "action_rate", "joint_limits", "contact_slip"):
    base = cfg.rewards[name].func.__wrapped__

    def ramped(env, _f=base, **kw):
      return _f(env, **kw) * reg2_scale(env)

    ramped.__name__ = f"{base.__name__}_reg2"
    cfg.rewards[name].func = ramped
  cfg.rewards["push_symmetry"] = RewardTermCfg(func=push_symmetry, weight=-1.0)
  return cfg


def microban_getup2_ppo_runner_cfg():
  cfg = microban_getup_ppo_runner_cfg()  # same experiment folder, so run 1 can be resumed
  cfg.max_iterations = 3500  # 500 -> 4000
  cfg.run_name = "getup2"
  return cfg
