import sys, torch, numpy as np
from dataclasses import asdict
import microban_rl
from microban_rl.getup_env_cfg import reference, ref_time
from microban_rl.symmetry import _MIRROR_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
cfg = load_env_cfg("Microban-GetUp", play=True); cfg.scene.num_envs = 64
ag = load_rl_cfg("Microban-GetUp"); env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
r = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device); r.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location=env.device)
pol = r.get_inference_policy(device=env.device)
robot = env.scene["robot"]; names = [n.split("/")[-1] for n in robot.joint_names]
with torch.inference_mode():
  env.reset(); obs = ve.get_observations(); ref = reference(env)
  segs = {"0-8 (to pike_1)": (0, 8.0), "8-10.4 (push)": (8.0, 10.4), "10.4-15 (late)": (10.4, 15.5)}
  acc = {k: [torch.zeros(64, 19, device=env.device), torch.zeros(64, device=env.device)] for k in segs}
  for s in range(790):
    obs, _, d, _ = ve.step(pol(obs)); t = ref_time(env)
    q = robot.data.joint_pos - ref.default
    for k, (a, b) in segs.items():
      m = ((t >= a) & (t < b)).float(); acc[k][0] += q * m[:, None]; acc[k][1] += m
for k, (sq, c) in acc.items():
  mq = (sq / c.clamp(min=1)[:, None]).cpu().numpy()
  row = []
  for l, (rr, sg) in _MIRROR_JOINTS.items():
    if l.startswith("left_"):
      dif = np.median(mq[:, names.index(l)] - sg * mq[:, names.index(rr)])
      row.append((abs(dif), l[5:], dif))
  row.sort(reverse=True)
  print(k, " | ".join(f"{n} {d:+.3f}" for _, n, d in row[:5]), f"| head {np.median(mq[:, names.index('head')]):+.3f}")
