"""Left/right mirror symmetry for Microban policies (rsl-rl Symmetry extension).

Used as ``symmetry_cfg["data_augmentation_func"]`` in the PPO config: every
mini-batch is augmented with its mirror image (the robot reflected through the
trunk x-z plane), which makes the learned policy left/right symmetric without
forbidding alternating gaits.

Joint pairs and signs were derived from the MJCF kinematics: mirroring a
right-joint rotation equals the left-joint rotation times the sign below.
The default pose in robot.py is mirror-symmetric, so the same map applies to
joint offsets relative to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlPpoAlgorithmCfg

# joint -> (mirrored joint, sign). Roll/yaw axes flip, pitch axes do not.
_MIRROR_JOINTS: dict[str, tuple[str, float]] = {"head": ("head", -1.0)}
for _side, _other in (("left", "right"), ("right", "left")):
  for _name, _sign in (
    ("shoulder_pitch", 1.0),
    ("shoulder_roll", -1.0),
    ("elbow", 1.0),
    ("hip_yaw", -1.0),
    ("hip_roll", -1.0),
    ("hip_pitch", 1.0),
    ("knee", 1.0),
    ("ankle_pitch", 1.0),
    ("ankle_roll", -1.0),
  ):
    _MIRROR_JOINTS[f"{_side}_{_name}"] = (f"{_other}_{_name}", _sign)

# Mirror of trunk-frame vectors through the x-z plane (y -> -y).
_POLAR = (1.0, -1.0, 1.0)  # velocities, gravity
_AXIAL = (-1.0, 1.0, -1.0)  # angular velocities

# Observation term name -> how it mirrors. Joint terms are resolved per env.
_TERM_SIGNS: dict[str, tuple[float, ...]] = {
  "base_ang_vel": _AXIAL,
  "projected_gravity": _POLAR,
  "base_lin_vel": _POLAR,
  "heading": (-1.0, 1.0),  # (sin, cos) of the heading error
  "root_height": (1.0,),
}
# Terms that are a permutation (+ signs) of their own entries.
_TERM_MAPS: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {
  # CPG phases [sin L, cos L, sin R, cos R]: the oscillators swap sides.
  "cpg_phase": ((2, 3, 0, 1), (1.0, 1.0, 1.0, 1.0)),
}
_JOINT_TERMS = {"joint_pos", "joint_vel"}
_ACTION_TERMS = {"actions"}


def mirror_joint_map(names: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
  """(index, sign) such that mirrored[:, i] = x[:, index[i]] * sign[i]."""
  idx = [names.index(_MIRROR_JOINTS[n][0]) for n in names]
  sign = [_MIRROR_JOINTS[n][1] for n in names]
  return (
    torch.tensor(idx, device=device, dtype=torch.long),
    torch.tensor(sign, device=device),
  )


_CACHE: dict[int, dict] = {}


def _maps(env) -> dict:
  """Per-group (index, sign) vectors such that mirrored = x[:, idx] * sign."""
  key = id(env)
  if key in _CACHE:
    return _CACHE[key]
  base = env.unwrapped
  device = base.device
  robot = base.scene["robot"]
  joint_idx, joint_sign = mirror_joint_map(list(robot.joint_names), device)
  # Single action term: either it knows its own mirror (CPG) or it is a plain
  # joint-position term over its target joints.
  (term_name,) = base.action_manager.active_terms
  term = base.action_manager.get_term(term_name)
  if hasattr(term, "mirror_map"):
    act_idx, act_sign = term.mirror_map()
  else:
    act_idx, act_sign = mirror_joint_map(list(term.target_names), device)

  om = base.observation_manager
  groups = {}
  for group, terms in om.active_terms.items():
    idx_parts, sign_parts, offset = [], [], 0
    for term, dim in zip(terms, om.group_obs_term_dim[group], strict=True):
      n = dim[-1]
      if term in _JOINT_TERMS:
        i, s = joint_idx, joint_sign
      elif term in _ACTION_TERMS:
        i, s = act_idx, act_sign
      elif term in _TERM_MAPS:
        i = torch.tensor(_TERM_MAPS[term][0], device=device)
        s = torch.tensor(_TERM_MAPS[term][1], device=device)
      elif term in _TERM_SIGNS:
        i = torch.arange(n, device=device)
        s = torch.tensor(_TERM_SIGNS[term], device=device)
      else:
        raise KeyError(f"No mirror rule for observation term '{term}'.")
      assert len(i) == n, f"term '{term}': expected {n} dims, mirror map has {len(i)}"
      idx_parts.append(i + offset)
      sign_parts.append(s)
      offset += n
    groups[group] = (torch.cat(idx_parts), torch.cat(sign_parts))
  _CACHE[key] = {"obs": groups, "act": (act_idx, act_sign)}
  return _CACHE[key]


def mirror_augmentation(env, obs: TensorDict | None, actions: torch.Tensor | None):
  """rsl-rl data_augmentation_func: returns [original; mirrored] obs and actions."""
  maps = _maps(env)
  obs_aug = None
  if obs is not None:
    mirrored = {}
    for group in obs.keys():
      idx, sign = maps["obs"][group]
      mirrored[group] = obs[group][:, idx] * sign
    mirrored_td = TensorDict(mirrored, batch_size=obs.batch_size, device=obs.device)
    obs_aug = torch.cat([obs, mirrored_td], dim=0)
  act_aug = None
  if actions is not None:
    idx, sign = maps["act"]
    act_aug = torch.cat([actions, actions[:, idx] * sign], dim=0)
  return obs_aug, act_aug


@dataclass
class SymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab PPO config plus rsl-rl's ``symmetry_cfg`` (mjlab does not expose it)."""

  symmetry_cfg: dict[str, str | bool | float] = field(
    default_factory=lambda: {
      "data_augmentation_func": "microban_rl.symmetry:mirror_augmentation",
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
    }
  )
