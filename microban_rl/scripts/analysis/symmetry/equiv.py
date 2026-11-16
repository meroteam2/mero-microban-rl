import sys, torch
from dataclasses import asdict
import microban_rl
from microban_rl.symmetry import mirror_augmentation, _maps
from microban_rl.getup_env_cfg import ref_time
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
cfg = load_env_cfg("Microban-GetUp", play=True); cfg.scene.num_envs = 64
cfg.events["reset_getup"].params["prone_params"] = {"yaw_range": (0.0, 0.0), "tilt_range": (0.0, 0.0), "joint_pos_range": (0.0, 0.0), "xy_range": (0.0, 0.0)}
ag = load_rl_cfg("Microban-GetUp"); env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
r = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device); r.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location=env.device)
pol = r.get_inference_policy(device=env.device)
mp = _maps(ve)
om = env.observation_manager
print("actor terms", list(zip(om.active_terms["actor"], [d[-1] for d in om.group_obs_term_dim["actor"]])))
with torch.inference_mode():
  env.reset(); obs = ve.get_observations()
  for s in range(560):
    o = obs["actor"]; idx, sg = mp["obs"]["actor"]
    if s % 70 == 0 or s in (480, 500, 520):
      mo = o[:, idx] * sg
      a = pol(obs); am_idx, am_s = mp["act"]
      obs_m, _ = mirror_augmentation(ve, obs, None); obs_m = obs_m[64:]
      a_m = pol(obs_m)
      sym_state = (mo - o).abs().max().item()
      eq = (a_m - a[:, am_idx] * am_s).abs().max().item()
      off = (o - mo).abs()[0]
      j = int(off.argmax())
      print(f"t {ref_time(env)[0].item():5.2f}  |M(obs)-obs| max {sym_state:.3f} (dim {j})  |pi(M obs) - M pi(obs)| max {eq:.3f}  |a| max {a.abs().max().item():.2f}")
    obs, _, _, _ = ve.step(pol(obs))
