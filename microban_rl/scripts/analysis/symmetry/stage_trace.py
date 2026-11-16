import sys, math, torch, numpy as np
from dataclasses import asdict
import microban_rl
from microban_rl.getup_env_cfg import getup_state, ref_time, reference, stand_ok, HOLD_S
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
N = 64
cfg = load_env_cfg("Microban-GetUp-4", play=True); cfg.scene.num_envs = N
ag = load_rl_cfg("Microban-GetUp-4"); env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
r = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device); r.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location=env.device)
pol = r.get_inference_policy(device=env.device)
torch.manual_seed(1)
with torch.inference_mode():
  env.reset(); obs = ve.get_observations(); ref = reference(env)
  steps = int(round((ref.t_end + HOLD_S + 0.5) / env.step_dt))
  min_gap = torch.full((N,), 9.0, device=env.device); ok_time = torch.zeros(N, device=env.device)
  print("  t    tilt   z     hands feet  CoM-toe(mm)  cp-toe(mm) stand_ok%")
  for s in range(steps):
    obs, _, d, ex = ve.step(pol(obs)); st = getup_state(env); t = ref_time(env)[0].item()
    if d.any(): print(f"  step {s}: {int(d.sum())} done, time_outs {int(ex['time_outs'].sum())}, env0 done {bool(d[0])}")
    four = st["hands_down"] & st["feet_down"] & (st["z"] >= 0.13)
    min_gap = torch.where(four, torch.minimum(min_gap, st["com_x"] - st["toe"]), min_gap)
    ok = stand_ok(env); ok_time += ok.float() * env.step_dt
    if s % 25 == 0 and s * 0.02 > 8.0:
      print(f"{t:5.2f} {math.degrees(st['tilt'][0]):6.1f} {st['z'][0]:.3f}  {int(st['hands_down'][0])}     {int(st['feet_down'][0])}    {(st['com_x'][0]-st['toe'][0])*1e3:+7.1f}    {(st['cp'][0]-st['toe'][0])*1e3:+7.1f}   {ok.float().mean()*100:5.1f}")
  g = min_gap.cpu().numpy()*1e3
  print(f"min CoM-toe with hands+feet down (mm): median {np.median(g):+.1f}, p10 {np.percentile(g,10):+.1f}, p90 {np.percentile(g,90):+.1f}; envs <= 10 mm: {(g<=10).mean()*100:.0f}%")
  print(f"time in stand conditions per env (s): median {ok_time.median().item():.2f}, min {ok_time.min().item():.2f}")
