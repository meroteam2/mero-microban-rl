"""XL330 servo model with gear backlash, gain/torque and calibration randomization.

The real joint has play between the servo output (where the magnetic encoder
sits) and the link: XL330 gear train plus the horn / 3D-printed bracket fit.
Model, per joint, with a backlash gap of total width ``b`` (half gap h = b/2):

* The motor side (encoder position ``theta``) is treated as massless next to
  the link, so with the PD loop on ``theta`` it sits on the target unless the
  link blocks it:  ``theta = clip(target, q - h, q + h)``.
* Torque reaches the link only when the gap is closed (``|target - q| > h``):
  ``tau = kp * deadzone(target - q, h) - kd * qdot``, clamped to the effort
  limit and to the direction the gap is closed in (the motor pushes the link
  through the play but never pulls on it).
* Inside the gap the link is free (only the MJCF joint friction / back-EMF
  damping act) and the encoder does not see it: the policy observes ``theta``.

Per env and joint, drawn at every reset:
* ``backlash_range``: total gap b (rad).
* ``kp_scale_range`` / ``effort_scale_range``: P gain and torque limit factors
  (supply voltage, register rounding, motor-to-motor spread).
* ``offset_range``: calibration error c. The servo's zero is off by c: it
  reports ``theta + c`` and drives its own reading to the target, i.e. the
  physical target is ``target - c``.

With all ranges 0 / 1 this is exactly the ideal PD of ``robot.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BacklashPdActuatorCfg(IdealPdActuatorCfg):
  backlash_range: tuple[float, float] = (0.0, 0.0)
  """Total backlash gap per joint (rad)."""
  kp_scale_range: tuple[float, float] = (1.0, 1.0)
  effort_scale_range: tuple[float, float] = (1.0, 1.0)
  offset_range: tuple[float, float] = (0.0, 0.0)
  """Calibration (zero) error per joint (rad)."""

  def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> BacklashPdActuator:
    return BacklashPdActuator(self, entity, target_ids, target_names)


def _uniform(rng: tuple[float, float], like: torch.Tensor) -> torch.Tensor:
  lo, hi = rng
  return lo + (hi - lo) * torch.rand_like(like)


class BacklashPdActuator(IdealPdActuator[BacklashPdActuatorCfg]):
  """Ideal PD on the encoder (motor-side) angle, transmitted through a dead zone."""

  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    n, j = data.nworld, len(self._target_names)
    self.half_gap = torch.zeros(n, j, device=device)
    self.offset = torch.zeros(n, j, device=device)
    self.sensed_pos: torch.Tensor | None = None  # encoder reading (theta + offset), target order
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if not hasattr(self, "half_gap"):
      return
    if env_ids is None:
      env_ids = slice(None)
    c = self.cfg
    ref = self.half_gap[env_ids]
    self.half_gap[env_ids] = 0.5 * _uniform(c.backlash_range, ref)
    self.offset[env_ids] = _uniform(c.offset_range, ref)
    assert self.stiffness is not None and self.force_limit is not None
    assert self.default_stiffness is not None and self.default_force_limit is not None
    self.stiffness[env_ids] = self.default_stiffness[env_ids] * _uniform(c.kp_scale_range, ref)
    self.force_limit[env_ids] = self.default_force_limit[env_ids] * _uniform(c.effort_scale_range, ref)
    if self.sensed_pos is not None:
      self.sensed_pos[env_ids] = float("nan")  # refreshed on the next compute

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    h = self.half_gap
    target = cmd.position_target - self.offset  # physical target
    err = target - cmd.pos
    dz = torch.where(err > h, err - h, torch.where(err < -h, err + h, torch.zeros_like(err)))
    assert self.stiffness is not None and self.damping is not None and self.force_limit is not None
    tau = self.stiffness * dz + self.damping * (cmd.velocity_target - cmd.vel) + cmd.effort_target
    tau = torch.clamp(tau, -self.force_limit, self.force_limit)
    # Only push in the direction the gap is closed; free inside the gap.
    zero = torch.zeros_like(tau)
    tau = torch.where(dz > 0, tau.clamp(min=0.0), torch.where(dz < 0, tau.clamp(max=0.0), zero))
    self.sensed_pos = torch.clamp(target, cmd.pos - h, cmd.pos + h) + self.offset
    return tau


def backlash_actuator(asset: Entity) -> BacklashPdActuator | None:
  for act in asset.actuators:
    if isinstance(act, BacklashPdActuator):
      return act
  return None


def sensed_joint_pos(asset: Entity) -> torch.Tensor:
  """Encoder joint angles [n, n_joints]; the true q where there is no backlash actuator."""
  q = asset.data.joint_pos
  act = backlash_actuator(asset)
  if act is None:
    return q
  out = q.clone()
  ids = act.target_ids
  if act.sensed_pos is None:
    out[:, ids] = q[:, ids] + act.offset
    return out
  # Right after a reset the last compute is stale: use q (+ offset) until the next physics step.
  out[:, ids] = torch.where(torch.isnan(act.sensed_pos), q[:, ids] + act.offset, act.sensed_pos)
  return out


def joint_pos_rel_sensed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Like mjlab's joint_pos_rel, but the encoder (motor-side, miscalibrated) angle."""
  asset: Entity = env.scene[asset_cfg.name]
  jid = asset_cfg.joint_ids
  return (sensed_joint_pos(asset) - asset.data.default_joint_pos)[:, jid]
