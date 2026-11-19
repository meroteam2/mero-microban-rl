"""Smoke test for Microban-Walk: obs layout, floor DR, backlash, stepping with zero actions.

uv run python scripts/check_walk_env.py [--num-envs 64]
"""

from __future__ import annotations

import argparse

import torch

import microban_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from microban_rl.backlash import backlash_actuator, sensed_joint_pos


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=90)
  p.add_argument("--task", default="Microban-Walk-DR")
  a = p.parse_args()

  cfg = load_env_cfg(a.task)
  cfg.scene.num_envs = a.num_envs
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  robot = env.scene["robot"]
  mj = env.sim.mj_model
  print("step_dt", env.step_dt, "physics dt", mj.opt.timestep)
  print("obs:", {k: tuple(v.shape) for k, v in obs.items()})
  om = env.observation_manager
  for g in ("actor", "critic"):
    print(g, list(zip(om.active_terms[g], [d[-1] for d in om.group_obs_term_dim[g]])))
  term = env.action_manager.get_term("joint_pos")
  print("action joints:", list(term.target_names))

  ids, names = robot.find_geoms(r"^col_.*$")
  gids = robot.indexing.geom_ids[ids]
  print(f"collision geoms: {len(names)}  e.g. {names[:3]} ... {names[-2:]}")
  print("robot priority:", sorted({int(mj.geom_priority[g]) for g in gids.tolist()}))
  fr = env.sim.model.geom_friction[:, gids, 0]
  sr = env.sim.model.geom_solref[:, gids]
  same = bool((fr.max(1).values - fr.min(1).values).abs().max() < 1e-6)
  print(f"friction {fr.min():.3f}..{fr.max():.3f} (one value per env: {same})")
  print(f"solref timeconst {sr[..., 0].min():.4f}..{sr[..., 0].max():.4f}  dampratio {sr[..., 1].min():.3f}..{sr[..., 1].max():.3f}")

  act = backlash_actuator(robot)
  assert act is not None
  gap = (2 * act.half_gap).rad2deg()
  print(f"backlash deg {gap.min():.2f}..{gap.max():.2f}  offset deg {act.offset.rad2deg().min():.2f}..{act.offset.rad2deg().max():.2f}")
  print(f"kp {act.stiffness.min():.3f}..{act.stiffness.max():.3f}  effort {act.force_limit.min():.3f}..{act.force_limit.max():.3f}")

  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  falls = 0
  for _ in range(a.steps):
    obs, rew, term_, trunc, extras = env.step(zero)
    d = (sensed_joint_pos(robot)[:, act.target_ids] - robot.data.joint_pos[:, act.target_ids] - act.offset).abs()
    # The encoder value is from the start of the last physics substep (one substep behind q).
    slack = robot.data.joint_vel[:, act.target_ids].abs() * env.sim.mj_model.opt.timestep * 2 + 1e-4
    assert torch.all(d <= act.half_gap + slack), float((d - act.half_gap - slack).max())
    falls += int(term_.sum())
  print(f"after {a.steps} steps of zero action: terminations {falls}, trunk z mean {robot.data.root_link_pos_w[:, 2].mean():.3f}")
  print("finite:", all(bool(torch.isfinite(v).all()) for v in obs.values()), " reward mean", float(rew.mean()))
  site_ids, _ = robot.find_sites(("right_foot", "left_foot"), preserve_order=True)
  print("foot site z", robot.data.site_pos_w[:, site_ids, 2].mean(0).tolist())
  for k, v in extras.get("log", {}).items():
    if k.startswith("Episode_Reward"):
      print(" ", k, float(v))
  env.close()


if __name__ == "__main__":
  main()
