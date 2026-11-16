"""Recompute the three bridge poses deterministically and save qpos (editor scene order)."""
import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
out = {}
# 1) pike_push: hands on floor, CoM as far back toward the toes as possible (pike family, from pike_1)
r = solve(lambda f: (f['com'] - f['toe']) * 1e2, lambda f: [1.0 - 1e3 * f['hand']], seeds=[pseed], n=0)
out['pike_push'] = build(r.x)
# 2) fold: hands lifted, CoM over the soles (10 mm margin), legs straight; start from pike_push
r = solve(lambda f: f['hand'] * 1e2, lambda f: [1e2 * (f['com'] - f['heel'] - 0.010), 1e2 * (f['toe'] - 0.010 - f['com'])], seeds=[r.x], n=0)
out['fold'] = build(r.x)
# 3) squat_deep: knee bent, hip -1.54, CoM over sole middle, soles flat (roll-up start)
def P(y): return build(np.array([y[2], -0.5, -0.35, -1.54, y[0], y[1], 0.087, -0.087]))
def obj(y):
  f = feats(P(y)); return 1e4 * (f['com'] - (f['heel'] + f['toe']) / 2) ** 2 + 1e5 * f['flat'] ** 2 + 1e3 * max(0, -f['other'])
best = min((minimize(obj, y0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-5, fatol=1e-10)) for y0 in ([1.7, -0.7, 0.5], [1.5, -0.6, 0.8])), key=lambda r: r.fun)
out['squat_deep'] = P(best.x)
for k, q in out.items():
  q[0], q[1] = 0.34, 0.0
  f = feats(q); tau, til, qe, drift = ST.hold(q); j = int(np.argmax(abs(tau)))
  print(f"{k:11s} tilt {ST.tilt(q):5.1f}  trunk z {q[2]:.3f}  CoM-toe {(f['com']-f['toe'])*1e3:+5.1f} mm  CoM-heel {(f['com']-f['heel'])*1e3:+5.1f}  hand z {f['hand']*1e3:5.1f} mm  sole flat {f['flat']*1e3:.1f} mm  other parts >= {f['other']*1e3:.1f} mm  hold {abs(tau).max():.3f} Nm ({names[j]}), tilt after 1.5 s {til:.1f}")
  print('            ' + ', '.join(f"{n.replace('left_','L_')} {q[qa[J[n]]]:+.2f}" for n in names if n.startswith('left_')))
  np.save(f'{sys.argv[1]}/v6_{k}.npy', q)
