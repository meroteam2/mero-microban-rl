"""Measure ground contact normal forces of a trained Microban policy.

Records every physics substep (200 Hz, not just policy steps) so that short
impact spikes are not missed. Per body part it reports mean / max / top-1%
normal force (also as multiples of body weight), splits contact time into the
impact window (first 10 ms after touchdown) and sustained contact, and reports
the downward speed of the contact point at touchdown.

  uv run python scripts/contact_forces.py Microban-Crawl \
      --checkpoint-file logs/rsl_rl/microban_crawl/<run>/model_1499.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

import microban_rl  # noqa: F401
from microban_rl.crawl_env_cfg import GROUND_CONTACT_SENSOR, robot_weight
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

# Net contact wrench per body, global frame.
FORCE_SENSOR = ContactSensorCfg(
  name="force_probe",
  primary=GROUND_CONTACT_SENSOR.primary,
  secondary=GROUND_CONTACT_SENSOR.secondary,
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
)
IMPACT_SUBSTEPS = 2  # 2 x 5 ms = first 10 ms of a contact


@dataclass(frozen=True)
class Args:
  task_id: tyro.conf.Positional[str]
  checkpoint_file: str
  num_envs: int = 256
  seconds: float = 10.0
  solver_iterations: int | None = None
  """Override the constraint solver iteration limit (training uses 10)."""


def main(
  task_id: str,
  checkpoint_file: str,
  num_envs: int,
  seconds: float,
  solver_iterations: int | None,
) -> None:
  device = "cuda:0"
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs
  sensors = tuple(env_cfg.scene.sensors or ())
  if not any(c.name == GROUND_CONTACT_SENSOR.name for c in sensors):
    sensors += (GROUND_CONTACT_SENSOR,)
  env_cfg.scene.sensors = sensors + (FORCE_SENSOR,)
  if solver_iterations is not None:
    env_cfg.sim.mujoco.iterations = solver_iterations
    env_cfg.sim.mujoco.ls_iterations = 2 * solver_iterations
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), device=device)
  runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  probe = env.scene["force_probe"]
  point = env.scene["ground_contact"]
  names = list(probe.primary_names)
  body_ids = [robot.body_names.index(n) for n in names]
  weight = robot_weight(env)

  fz_log, found_log, vz_log = [], [], []

  def record_substep() -> None:
    fz_log.append(probe.data.force[..., 2].abs().clone())
    found_log.append((probe.data.found > 0).clone())
    vel = robot.data.body_link_vel_w[:, body_ids]
    r = point.data.pos - robot.data.body_link_pos_w[:, body_ids]
    vz_log.append((vel[..., :3] + torch.cross(vel[..., 3:], r, dim=-1))[..., 2].clone())

  original = env.metrics_manager.compute_substep

  def patched(*a, **k):
    out = original(*a, **k)
    record_substep()
    return out

  env.metrics_manager.compute_substep = patched

  obs = vec_env.get_observations()
  with torch.inference_mode():
    for _ in range(int(seconds / env.step_dt)):
      obs, _, _, _ = vec_env.step(policy(obs))

  fz = torch.stack(fz_log).cpu().numpy()  # [T, N, P]
  found = torch.stack(found_log).cpu().numpy()
  vz = torch.stack(vz_log).cpu().numpy()
  T = fz.shape[0]

  # Substeps since touchdown, per (env, body).
  since = np.zeros_like(fz, dtype=np.int32)
  run = np.zeros(fz.shape[1:], dtype=np.int32)
  for t in range(T):
    run = np.where(found[t], run + 1, 0)
    since[t] = run
  touchdown = since == 1
  impact = (since >= 1) & (since <= IMPACT_SUBSTEPS)
  sustained = since > IMPACT_SUBSTEPS

  total = (fz * found).sum(-1)
  print(f"\n=== contact forces: {task_id}, {num_envs} envs x {seconds:.0f}s, "
        f"200 Hz, solver iterations {env_cfg.sim.mujoco.iterations} ===")
  print(f"body weight W = {weight:.2f} N | total ground force mean "
        f"{total.mean():.2f} N ({total.mean() / weight:.2f} W), p99 "
        f"{np.percentile(total, 99):.2f} N ({np.percentile(total, 99) / weight:.2f} W)")
  hdr = (f"{'body':34s} {'contact%':>8s} {'mean':>11s} {'p99':>11s} {'max':>11s}"
         f" {'impact p99':>11s} {'sustain p99':>11s} {'top1% in impact':>15s}"
         f" {'touchdown vz':>13s}")
  print(hdr)
  rows = []
  for i, n in enumerate(names):
    c = found[..., i]
    if c.mean() < 0.005:  # touches the ground <0.5% of the time: too few samples
      continue
    f = fz[..., i][c]
    p99 = np.percentile(f, 99)
    top = fz[..., i] >= p99
    top_in_impact = (top & impact[..., i]).sum() / max((top & c).sum(), 1)
    fi = fz[..., i][impact[..., i]]
    fs = fz[..., i][sustained[..., i]]
    td = -vz[..., i][touchdown[..., i]]
    rows.append((n, c.mean(), f.mean(), p99, f.max(),
                 np.percentile(fi, 99) if fi.size else 0.0,
                 np.percentile(fs, 99) if fs.size else 0.0, top_in_impact,
                 np.median(td) if td.size else 0.0))
  for n, pct, mean, p99, mx, ip99, sp99, tii, td in sorted(rows, key=lambda r: -r[3]):
    print(f"{n:34s} {pct * 100:7.1f}% "
          f"{mean:5.2f}N/{mean / weight:4.2f}W {p99:5.2f}N/{p99 / weight:4.2f}W "
          f"{mx:5.2f}N/{mx / weight:4.2f}W {ip99 / weight:9.2f}W "
          f"{sp99 / weight:10.2f}W {tii * 100:14.0f}% {td:10.3f}m/s")
  print("\n(W = body weight. 'top1% in impact' = share of each body's top-1% force "
        f"samples that fall in the first {IMPACT_SUBSTEPS * 5} ms of a contact; "
        "touchdown vz = median downward speed of the contact point at touchdown.)")


if __name__ == "__main__":
  main(**asdict(tyro.cli(Args)))
