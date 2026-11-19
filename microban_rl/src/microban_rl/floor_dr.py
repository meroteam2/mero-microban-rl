"""Floor domain randomization: friction and contact softness per env.

MuJoCo combines two geoms' contact parameters by max (friction) / solmix
(solref, solimp) unless their priorities differ, in which case the higher
priority geom's parameters are used as-is. So every robot collision geom gets
``priority=1`` (the floor plane stays at 0) and the randomized values are
written to the robot's collision geoms with one shared draw per env: every
robot-floor contact in an env then sees the same "floor". Robot self-contacts
are between equal priorities and keep the usual mixing.

Drawn at every reset:
* tangential friction mu (PLA on hard floor ~0.4 .. rubber pad / carpet ~1.2)
* solref (timeconst, dampratio): hard floor (0.012 s) .. soft mat / carpet
  (0.04 s); timeconst >= 2 * physics dt (1/180 s for Microban-Walk) keeps contacts stable.
"""

from __future__ import annotations

import mujoco
from mjlab.envs.mdp.dr import _core
from mjlab.envs.mdp.dr.geom import geom_friction
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from microban_rl.robot import get_spec

COLLISION_GROUP = 3  # robot.xml: <default class="collision"><geom group="3"/>
COLLISION_GEOM_EXPR = r"^col_.*$"

FLOOR_FRICTION = (0.4, 1.2)
FLOOR_TIMECONST = (0.012, 0.04)
FLOOR_DAMPRATIO = (0.7, 1.3)


def get_spec_floor_dr() -> mujoco.MjSpec:
  """robot.py's spec with named, priority-1 collision geoms."""
  spec = get_spec()
  counts: dict[str, int] = {}
  for g in spec.geoms:
    if g.group != COLLISION_GROUP or g.contype == 0 and g.conaffinity == 0:
      continue
    body = g.parent.name
    i = counts.get(body, 0)
    counts[body] = i + 1
    g.name = f"col_{g.name or body + '_' + str(i)}"
    g.priority = 1
  return spec


@requires_model_fields("geom_solref")
def geom_solref(env, env_ids, ranges, asset_cfg: SceneEntityCfg, distribution="uniform", operation="abs",
                axes=None, shared_random: bool = False) -> None:
  _core._randomize_model_field(
    env, env_ids, "geom_solref", entity_type="geom", ranges=ranges, distribution=distribution,
    operation=operation, asset_cfg=asset_cfg, axes=axes, shared_random=shared_random,
    default_axes=[0, 1], valid_axes=[0, 1],
  )


def floor_events() -> dict[str, EventTermCfg]:
  geoms = SceneEntityCfg("robot", geom_names=(COLLISION_GEOM_EXPR,))
  return {
    "floor_friction": EventTermCfg(
      func=geom_friction, mode="reset",
      params={"asset_cfg": geoms, "ranges": FLOOR_FRICTION, "shared_random": True},
    ),
    "floor_solref": EventTermCfg(
      func=geom_solref, mode="reset",
      params={"asset_cfg": geoms, "ranges": {0: FLOOR_TIMECONST, 1: FLOOR_DAMPRATIO}, "shared_random": True},
    ),
  }
