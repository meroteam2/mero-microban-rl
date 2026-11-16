"""Per checkpoint (prone starts, play cfg, 256 envs, one episode): per segment
- left/right difference per joint pair (median over envs of |segment-mean q_L - s q_R|), hip_yaw explicitly
- share of alive time each joint is at the torque limit (|tau| >= 0.98 x 0.64 Nm), top joints."""
import os, sys, json, torch, numpy as np
from dataclasses import asdict
import microban_rl
from microban_rl.getup_env_cfg import reference, ref_time, HOLD_S
from microban_rl.robot import XL330_EFFORT_LIMIT
from microban_rl.symmetry import _MIRROR_JOINTS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
N = 256
cfg = load_env_cfg(os.environ.get("TASK", "Microban-GetUp-2"), play=True); cfg.scene.num_envs = N
ag = load_rl_cfg(os.environ.get("TASK", "Microban-GetUp-2")); env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); ve = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
robot = env.scene["robot"]; names = [n.split("/")[-1] for n in robot.joint_names]
pairs = [(l, names.index(l), names.index(r), s) for l, (r, s) in _MIRROR_JOINTS.items() if l.startswith("left_")]
out = {}
for ck in sys.argv[1:]:
  torch.manual_seed(1)
  r = MjlabOnPolicyRunner(ve, asdict(ag), device=env.device); r.load(ck, load_cfg={"actor": True}, strict=True, map_location=env.device)
  pol = r.get_inference_policy(device=env.device)
  with torch.inference_mode():
    env.reset(); obs = ve.get_observations(); ref = reference(env)
    seg = {"early": (-9, ref.kf_t["pike_1"]), "push": (ref.kf_t["pike_1"], ref.kf_t["squat_deep"]), "late": (ref.kf_t["squat_deep"], 99)}
    sq = {k: torch.zeros(N, 19, device=env.device) for k in seg}; sc = {k: torch.zeros(N, device=env.device) for k in seg}
    sat = {k: torch.zeros(19, device=env.device) for k in seg}
    alive = torch.ones(N, dtype=torch.bool, device=env.device)
    steps = int(round((ref.t_end + HOLD_S + 0.5) / env.step_dt))
    for s in range(steps):
      obs, _, d, _ = ve.step(pol(obs)); alive &= ~d.bool()
      t = ref_time(env); q = robot.data.joint_pos - ref.default
      at = (robot.data.actuator_force.abs() >= 0.98 * XL330_EFFORT_LIMIT).float()
      for k, (a, b) in seg.items():
        m = (alive & (t >= a) & (t < b)).float()
        sq[k] += q * m[:, None]; sc[k] += m; sat[k] += (at * m[:, None]).sum(0)
  res = {}
  for k in seg:
    mq = (sq[k] / sc[k].clamp(min=1)[:, None]).cpu().numpy(); has = sc[k].cpu().numpy() > 0
    diffs = {l[5:]: float(np.median(np.abs(mq[has, i] - s_ * mq[has, j]))) for l, i, j, s_ in pairs}
    share = (sat[k] / sc[k].sum().clamp(min=1)).cpu().numpy()
    top = sorted(((float(share[i]), names[i]) for i in range(19)), reverse=True)[:5]
    res[k] = {"pair_diff": diffs, "sat_any_joint": float(share.sum()), "sat_top": top}
  out[ck.split("/")[-1]] = res
  print(ck.split("/")[-1])
  for k, v in res.items():
    pd = sorted(v["pair_diff"].items(), key=lambda x: -x[1])
    print(f"  {k:5s} hip_yaw {v['pair_diff']['hip_yaw']:.3f} | worst " + ", ".join(f"{a} {b:.3f}" for a, b in pd[:3])
          + " | at torque limit: " + ", ".join(f"{n} {s*100:.1f}%" for s, n in v["sat_top"] if s > 0.0005))
print("JSON " + json.dumps(out))
