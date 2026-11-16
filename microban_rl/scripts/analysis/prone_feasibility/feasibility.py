"""Kinematic/static check of the hands->feet weight transfer (pike_1 -> crouch). Symmetric poses."""
import sys, math, numpy as np, joblib, mujoco
from scipy.optimize import minimize
sys.path.insert(0, sys.argv[1]); import static_torque as ST
m, d, qa, names = ST.m, ST.d, ST.qa, ST.names
J = {n: i for i, n in enumerate(names)}
kf = {k['name']: k for k in joblib.load('/home/mero/mero/keyframe_editor/keyframes/getup_1111.lz4')['keyframes']}
def kq(n):
  q = np.asarray(kf[n]['qpos'], dtype=np.float64).copy(); q[qa] = kf[n]['motor_pos']; return ST.place(q)
corner = [m.geom(f'{s}_foot_collision_{i}').id for s in ('left', 'right') for i in (3, 4, 5, 6)]
hand_b = [m.body('radius').id, m.body('radius_2').id]
tips = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER and m.geom_size[g][0] < 0.007]
SYM = ['shoulder_pitch', 'elbow', 'hip_pitch', 'knee', 'ankle_pitch']
SKIP = {'foot', 'foot_2', 'radius', 'radius_2'}
base = kq('crouch')
def build(x):
  q = base.copy(); q[3:7] = [math.cos(x[0] / 2), 0, math.sin(x[0] / 2), 0]
  for k, v in zip(SYM, x[1:6]): q[qa[J['left_' + k]]] = v; q[qa[J['right_' + k]]] = v
  for k, v in zip(('hip_roll', 'ankle_roll'), x[6:8]): q[qa[J['left_' + k]]] = v; q[qa[J['right_' + k]]] = -v
  q[2] = 0.3; q[2] -= min(ST.lowest_by_body(q).values()); return q
def feats(q):
  lb = ST.lowest_by_body(q)  # runs forward
  cz = np.array([d.geom_xpos[g][2] for g in corner]); cx = np.array([d.geom_xpos[g][0] for g in corner])
  hx = np.mean([d.geom_xpos[g][0] for g in tips])
  return dict(flat=np.ptp(cz), foot=max(lb['foot'], lb['foot_2']), hand=min(lb['radius'], lb['radius_2']),
              other=min(v for b, v in lb.items() if b not in SKIP), heel=cx.min(), toe=cx.max(), hx=hx,
              com=d.subtree_com[1][0], comz=d.subtree_com[1][2])
bounds = [(-0.2, 2.2)] + [tuple(np.array(m.jnt_range[m.joint('left_' + k).id]) + [0.03, -0.03]) for k in SYM]
bounds[3] = (bounds[3][0], 0.0)
bounds += [(-0.3, 0.3), (-0.3, 0.3)]
rng = np.random.default_rng(0)
def solve(obj, extra, seeds=(), n=40):
  def con(x):
    f = feats(build(x)); return np.array([1e3 * (0.0005 - f['flat']), 1e3 * (0.001 - f['foot']), 1e3 * f['other']] + extra(f))
  best = None
  for x0 in [np.array(s) for s in seeds] + [np.array([rng.uniform(a, b) for a, b in bounds]) for _ in range(n)]:
    x0 = np.clip(x0, *zip(*bounds))
    r = minimize(lambda x: obj(feats(build(x))), x0, method='SLSQP', bounds=bounds, constraints=[{'type': 'ineq', 'fun': con}], options=dict(maxiter=400))
    if con(r.x).min() > -0.3 and (best is None or r.fun < best.fun): best = r
  return best
def desc(x): return ', '.join(f'{k} {v:+.2f}' for k, v in zip(SYM, x[1:6])) + f', root pitch {x[0]:+.2f}, hip_roll(L) {x[6]:+.2f}, ankle_roll(L) {x[7]:+.2f}'
pk = kq('pike_1'); pseed = [2 * math.atan2(pk[5], pk[3])] + [pk[qa[J['left_' + k]]] for k in SYM] + [pk[qa[J['left_hip_roll']]], pk[qa[J['left_ankle_roll']]]]
f = feats(pk); print(f"pike_1: CoM {(f['com']-f['toe'])*1e3:+.0f} mm ahead of toes, hands {(f['hx']-f['toe'])*1e3:.0f} mm ahead of toes, tilt {ST.tilt(pk):.0f}")
f = feats(kq('crouch')); print(f"crouch: CoM {(f['com']-f['heel'])*1e3:.0f} mm from heel (sole {(f['toe']-f['heel'])*1e3:.0f} mm), tilt {ST.tilt(kq('crouch')):.0f}")
# A) CoM over the soles -> how low can the hands go
for mg in (0.010, 0.0):
  r = solve(lambda f: f['hand'] * 1e2, lambda f: [1e2 * (f['com'] - f['heel'] - mg), 1e2 * (f['toe'] - mg - f['com'])], seeds=[[0.76, -0.76, 0.0, -1.52, 2.28, -1.52, 0.1, -0.09]])
  q = build(r.x); f = feats(q)
  print(f"A) CoM over soles, margin {mg*1e3:.0f} mm: lowest hand tip {f['hand']*1e3:.1f} mm above floor; tilt {ST.tilt(q):.0f} | {desc(r.x)}")
# B) hands on the floor -> how far back can the CoM go
r = solve(lambda f: (f['com'] - f['toe']) * 1e2, lambda f: [1.0 - 1e3 * f['hand']], seeds=[pseed, [0.76, -0.76, 0.0, -1.52, 2.28, -1.52, 0.1, -0.09], [1.2, -1.6, 0.0, -1.5, 2.2, -1.4, 0.1, -0.09]])
q = build(r.x); f = feats(q); e = f['com'] - f['toe']; h = f['comz']; mass = m.body_subtreemass[1]
print(f"B) hands on floor: CoM at best {e*1e3:+.1f} mm ahead of the toes, CoM height {h:.3f} m, tilt {ST.tilt(q):.0f}, hands {(f['hx']-f['toe'])*1e3:.0f} mm ahead of toes | {desc(r.x)}")
np.save(f'{sys.argv[1]}/B.npy', q)
tau, til, qe, drift = ST.hold(q)
print(f"   static hold (4 points): torque max {abs(tau).max():.3f} Nm ({abs(tau).max()/ST.TL*100:.0f}%): " + ', '.join(f'{names[j]} {tau[j]:+.3f}' for j in np.argsort(-abs(tau))[:4]) + f"; drift {drift:.3f}")
v = math.sqrt(9.81 / h) * max(e, 0)
print(f"   to carry the CoM over the toes after the hands lift: backward CoM speed >= {v:.2f} m/s (mass {mass:.3f} kg) -> mean push {mass*v/0.1:.1f} N over 0.1 s / {mass*v/0.2:.1f} N over 0.2 s")
d.qpos[:] = q; mujoco.mj_forward(m, d); tot = 0
for b, s in ((m.body('radius').id, 'right'), (m.body('radius_2').id, 'left')):
  g = [t for t in tips if m.geom_bodyid[t] == b][0]; jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, None, d.geom_xpos[g], b)
  Jh = jp[[0, 2]][:, [m.jnt_dofadr[m.joint(f'{s}_{k}').id] for k in ('shoulder_pitch', 'shoulder_roll', 'elbow')]]
  best = 0
  for fz in np.linspace(0, 15, 31):  # floor pushes up (fz>=0) and back (fx<0 on robot)
    for fx in np.linspace(0, 20, 201):
      if fx <= 0.6 * fz + 1e-9 and np.all(abs(Jh.T @ np.array([fx, fz])) <= ST.TL): best = max(best, fx)
  tot += best
print(f"   max backward push from both hands at the torque limit (friction 0.6): {tot:.1f} N")
