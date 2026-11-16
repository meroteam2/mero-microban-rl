"""Central pattern generator (CPG) action for Microban.

Two phase oscillators, one per body side, drive the rhythmic joints:

  dphi/dt = 2 pi f                       (shared master phase)
  phi_L = phi - Delta/2,  phi_R = phi + Delta/2

so Delta is the left/right phase difference. It starts at pi (alternating) and
the policy can shift it by up to +-MAX_PHASE_SHIFT (0.2 rad); the command is
low-pass filtered so the phases never jump. (Any Delta != pi is a left/right
asymmetric rhythm; with +-pi/2 the fourth run settled at 270 deg.)

Limb phases (diagonal / contralateral coupling, as in a commando crawl):
  left leg  -> phi_L        right leg -> phi_R
  right arm -> phi_L        left arm  -> phi_R
i.e. each arm swings with the opposite leg. Every CPG joint j then follows

  q_j = q_default_j + o_j + a_j sin(phi_limb) + b_j cos(phi_limb) + r_j

The (a_j, b_j) pair sets that joint's amplitude sqrt(a^2 + b^2) *and* its phase
lag inside the limb cycle, so e.g. knee vs. hip timing (or a different arm/leg
lag than the diagonal default) is learned. Non-rhythmic joints (head, hip
yaw/roll, ankle roll) only get the residual r_j.

Policy action layout (raw policy outputs, clipped to [-2, 2] by the runner):
  [0]        frequency            f = clamp(1.5 + 1.0 u, 0.8, 3.0) Hz
  [1]        L/R phase shift      Delta = pi + clamp(u, -1, 1) * 0.2
  [2:14]     a_j  (CPG joints)    * AMP_SCALE
  [14:26]    b_j  (CPG joints)    * AMP_SCALE
  [26:38]    o_j  (CPG joints)    * OFFSET_SCALE
  [38:57]    r_j  (all joints)    * residual scale (small on CPG joints)

Each joint's oscillation amplitude sqrt(a^2 + b^2) is scaled down so that its
peak target speed, amplitude * 2 pi f, stays below MAX_CPG_JOINT_SPEED (80% of
the XL330 no-load speed; the rest is headroom for the residual and for torque).

Joint targets are clamped to the soft joint limits of robot.py and go through
the same robot.py position actuators (same torque limit) as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from microban_rl.robot import XL330_NO_LOAD_SPEED
from microban_rl.symmetry import mirror_joint_map

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Rhythmic joints and the oscillator ("L" or "R") that drives each of them.
CPG_JOINTS: dict[str, str] = {
  "left_hip_pitch": "L",
  "left_knee": "L",
  "left_ankle_pitch": "L",
  "right_hip_pitch": "R",
  "right_knee": "R",
  "right_ankle_pitch": "R",
  "left_shoulder_pitch": "R",
  "left_shoulder_roll": "R",
  "left_elbow": "R",
  "right_shoulder_pitch": "L",
  "right_shoulder_roll": "L",
  "right_elbow": "L",
}

FREQ_CENTER = 1.5
FREQ_GAIN = 1.0
# Floor raised from 0.2 Hz: the second CPG run parked at the floor (a 5 s cycle)
# with saturated amplitudes, sweeping the limbs in the air without moving.
FREQ_RANGE = (0.8, 3.0)
MAX_PHASE_SHIFT = 0.2
PHASE_SHIFT_TAU_S = 0.3  # low-pass on the commanded L/R phase difference
AMP_SCALE = 0.3
OFFSET_SCALE = 0.4
RESIDUAL_SCALE_CPG = 0.15
RESIDUAL_SCALE_OTHER = 0.5
RAW_CLIP = 2.0  # matches the runner's clip_actions
MAX_CPG_JOINT_SPEED = 0.8 * XL330_NO_LOAD_SPEED  # 10.3 rad/s


@dataclass(kw_only=True)
class CpgActionCfg(ActionTermCfg):
  entity_name: str = "robot"

  def build(self, env: ManagerBasedRlEnv) -> CpgAction:
    return CpgAction(self, env)


class CpgAction(ActionTerm):
  cfg: CpgActionCfg
  _entity: Entity

  def __init__(self, cfg: CpgActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    names = list(self._entity.joint_names)
    self.joint_names = names
    self.cpg_names = list(CPG_JOINTS)
    dev = self.device
    self._cpg_ids = torch.tensor([names.index(n) for n in self.cpg_names], device=dev)
    self._cpg_is_right = torch.tensor(
      [CPG_JOINTS[n] == "R" for n in self.cpg_names], device=dev
    )
    n_cpg, n_all = len(self.cpg_names), len(names)
    self._action_dim = 2 + 3 * n_cpg + n_all
    self._res_scale = torch.full((n_all,), RESIDUAL_SCALE_OTHER, device=dev)
    self._res_scale[self._cpg_ids] = RESIDUAL_SCALE_CPG

    n = self.num_envs
    self._raw = torch.zeros(n, self._action_dim, device=dev)
    self.frequency = torch.full((n,), FREQ_CENTER, device=dev)
    self.phase_shift_cmd = torch.full((n,), math.pi, device=dev)
    self.phase_shift = torch.full((n,), math.pi, device=dev)
    self.phase = torch.zeros(n, device=dev)
    self._a = torch.zeros(n, n_cpg, device=dev)
    self._b = torch.zeros(n, n_cpg, device=dev)
    self._o = torch.zeros(n, n_cpg, device=dev)
    self._res = torch.zeros(n, n_all, device=dev)
    self._targets = self._entity.data.default_joint_pos.clone()

  # ActionTerm interface.

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    u = actions.clamp(-RAW_CLIP, RAW_CLIP)
    k = len(self.cpg_names)
    self.frequency = (FREQ_CENTER + FREQ_GAIN * u[:, 0]).clamp(*FREQ_RANGE)
    self.phase_shift_cmd = math.pi + u[:, 1].clamp(-1.0, 1.0) * MAX_PHASE_SHIFT
    a = u[:, 2 : 2 + k] * AMP_SCALE
    b = u[:, 2 + k : 2 + 2 * k] * AMP_SCALE
    # Peak target speed amplitude * 2 pi f must stay below the motor limit.
    max_amp = MAX_CPG_JOINT_SPEED / (2 * math.pi * self.frequency)
    scale = (max_amp[:, None] / torch.hypot(a, b).clamp(min=1e-6)).clamp(max=1.0)
    self._a, self._b = a * scale, b * scale
    self._o = u[:, 2 + 2 * k : 2 + 3 * k] * OFFSET_SCALE
    self._res = u[:, 2 + 3 * k :] * self._res_scale

  def apply_actions(self) -> None:
    dt = self._env.physics_dt
    alpha = dt / PHASE_SHIFT_TAU_S
    self.phase_shift = self.phase_shift + alpha * (self.phase_shift_cmd - self.phase_shift)
    self.phase = torch.remainder(self.phase + 2 * math.pi * self.frequency * dt, 2 * math.pi)

    phi_l, phi_r = self.side_phases()
    phi = torch.where(self._cpg_is_right, phi_r[:, None], phi_l[:, None])
    data = self._entity.data
    targets = data.default_joint_pos.clone() + self._res
    targets[:, self._cpg_ids] += self._o + self._a * torch.sin(phi) + self._b * torch.cos(phi)
    limits = data.soft_joint_pos_limits
    targets = targets.clamp(limits[..., 0], limits[..., 1])
    self._targets = targets
    self._entity.set_joint_position_target(targets - data.encoder_bias)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    n = self.phase[env_ids].shape[0]
    self.phase[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi
    self.phase_shift[env_ids] = math.pi
    self.phase_shift_cmd[env_ids] = math.pi
    self.frequency[env_ids] = FREQ_CENTER

  # Helpers.

  def side_phases(self) -> tuple[torch.Tensor, torch.Tensor]:
    half = 0.5 * self.phase_shift
    return self.phase - half, self.phase + half

  def mirror_map(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Index/sign vectors that mirror this term's actions left <-> right."""
    k = len(self.cpg_names)
    cpg_idx, cpg_sign = mirror_joint_map(self.cpg_names, self.device)
    all_idx, all_sign = mirror_joint_map(self.joint_names, self.device)
    one = torch.ones(1, device=self.device)
    idx = torch.cat([
      torch.tensor([0, 1], device=self.device),
      cpg_idx + 2,
      cpg_idx + 2 + k,
      cpg_idx + 2 + 2 * k,
      all_idx + 2 + 3 * k,
    ])
    # Frequency is shared; the L/R phase shift flips sign around pi.
    sign = torch.cat([one, -one, cpg_sign, cpg_sign, cpg_sign, all_sign])
    return idx, sign


def cpg_phase(
  env: ManagerBasedRlEnv, action_name: str = "cpg"
) -> torch.Tensor:
  """[sin phi_L, cos phi_L, sin phi_R, cos phi_R] of the CPG oscillators."""
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, CpgAction)
  phi_l, phi_r = term.side_phases()
  return torch.stack(
    [torch.sin(phi_l), torch.cos(phi_l), torch.sin(phi_r), torch.cos(phi_r)], dim=-1
  )
