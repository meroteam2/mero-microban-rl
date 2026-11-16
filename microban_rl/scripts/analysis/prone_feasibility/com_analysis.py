import sys, math, numpy as np, torch, joblib, mujoco
sys.path.insert(0, '/home/mero/mero/microban_rl/scripts')
import replay_keyframes as R
from mjlab.envs import ManagerBasedRlEnv
path = '/home/mero/mero/keyframe_editor/keyframes/getup_1111.lz4'
data = joblib.load(path)
# editor's own played trajectory (MuJoCo CPU, editor PD): trunk tilt over time
m = mujoco.MjModel.from_xml_path('/home/mero/mero/keyframe_editor/microban_scene.xml'); dd = mujoco.MjData(m)
tq = np.asarray(data['qpos']); tt = np.asarray(data['time'])
print('editor trajectory (physics playback in the editor):')
for t_ in (1.5, 2.5, 3.5, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5, 9.98):
  i = int(np.argmin(abs(tt - t_))); dd.qpos[:] = tq[i]; mujoco.mj_forward(m, dd)
  head = dd.xmat[m.body('trunk').id].reshape(3, 3)[:, 2]
  print(f'  t {tt[i]:.2f} tilt {math.degrees(math.acos(np.clip(head[2],-1,1))):6.1f}  z {tq[i,2]:.3f}')
a = R.Args(path=path, num_envs=4, noise=0.0)
env = ManagerBasedRlEnv(cfg=R.env_cfg(a), device='cuda:0'); robot = env.scene['robot']
names = list(robot.joint_names); dt = env.step_dt
targets, seq = R.editor_targets(data, names, dt)
settle = int(a.settle_s / dt); sched = np.concatenate([np.repeat(targets[:1], settle, 0), targets, np.repeat(targets[-1:], 50, 0)])
default = robot.data.default_joint_pos[0].cpu().numpy()
act = torch.tensor(sched - default, device=env.device, dtype=torch.float32)
mj = env.sim.mj_model
bid = {n: [i for i in range(mj.nbody) if mj.body(i).name.split('/')[-1] == n][0] for n in ('trunk', 'foot', 'foot_2', 'radius', 'radius_2')}
mass = torch.tensor(mj.body_mass, device=env.device)
geo = R.Geometry(env)
print('\nMJLab env0: CoM x vs feet/hands (heading +x is where the head points when prone)')
print('   t   tilt   CoMx  feet_x[min..max]  hands_x  contacts  tau_max  worst joint err')
with torch.inference_mode():
  env.reset()
  for t in range(len(sched)):
    env.step(act[t].expand(4, -1))
    st = t * dt - a.settle_s
    if st >= 6.9 and abs(st * 10 - round(st * 10)) < 1e-6 and round(st * 10) % 2 == 0 or abs(st-8.5)<1e-6:
      d = env.sim.data; xp = d.xipos[0]; com = (xp * mass[:, None]).sum(0) / mass.sum()
      # foot sole extents: geoms of foot bodies
      low = {k: v[0].item() for k, v in geo.lowest().items()}
      fx = [d.xpos[0, bid[b], 0].item() for b in ('foot', 'foot_2')]; hx = [d.xpos[0, bid[b], 0].item() for b in ('radius', 'radius_2')]
      head = R.trunk_angles(robot.data.root_link_quat_w[:1])['tilt'][0].item()
      err = (robot.data.joint_pos[0].cpu().numpy() - sched[t]); j = int(np.argmax(abs(err)))
      c = ' '.join(k for k, v in low.items() if v < R.CONTACT_Z)
      print(f'{st:5.1f} {math.degrees(head):6.1f} {com[0].item():6.3f}  feet {min(fx):.3f}..{max(fx):.3f}  hands {np.mean(hx):.3f}  {c:28s} {robot.data.actuator_force[0].abs().max().item():.2f}  {names[j].split("/")[-1]} {err[j]:+.2f}')
