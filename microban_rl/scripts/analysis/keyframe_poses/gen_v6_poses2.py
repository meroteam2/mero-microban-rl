import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
out = {'pike_push': np.load(f'{sys.argv[1]}/v6_pike_push.npy')}
mid = lambda f: (f['heel'] + f['toe']) / 2
w, x_, y_, z_ = out['pike_push'][3:7]
xp = [2 * math.atan2(y_, w)] + [out['pike_push'][qa[J['left_' + k]]] for k in SYM] + [out['pike_push'][qa[J['left_hip_roll']]], out['pike_push'][qa[J['left_ankle_roll']]]]
# fold: legs straight (knee <= -0.6), hands >= 5 mm up, CoM at the sole middle
bk = list(bounds); bounds[4] = (bounds[4][0], -0.6)
r = solve(lambda f: 1e4 * (f['com'] - mid(f)) ** 2, lambda f: [1e3 * f['hand'] - 5.0], seeds=[xp], n=20)
out['fold'] = build(r.x); bounds[:] = bk
# squat_deep: hip -1.54, knee >= 1.5, CoM at the sole middle, arms forward, bounded
bounds[3] = (-1.54, -1.54 + 1e-6); bounds[4] = (1.5, bounds[4][1])
r = solve(lambda f: 1e4 * (f['com'] - mid(f)) ** 2, lambda f: [1e3 * f['hand'] - 5.0], seeds=[[0.5, -0.5, -0.35, -1.54, 1.73, -0.67, 0.087, -0.087]], n=20)
out['squat_deep'] = build(r.x); bounds[:] = bk
for k, q in out.items():
  q[0], q[1] = 0.34, 0.0
  f = feats(q); tau, til, qe, drift = ST.hold(q); j = int(np.argmax(abs(tau)))
  print(f"{k:11s} tilt {ST.tilt(q):5.1f}  trunk z {q[2]:.3f}  CoM-toe {(f['com']-f['toe'])*1e3:+5.1f} mm  CoM-heel {(f['com']-f['heel'])*1e3:+5.1f}  hand z {f['hand']*1e3:5.1f}  flat {f['flat']*1e3:.1f}  other >= {f['other']*1e3:.1f} mm  hold {abs(tau).max():.3f} Nm ({names[j]}), tilt after 1.5 s {til:.1f}, drift {drift:.3f}")
  print('            ' + ', '.join(f"{n.replace('left_','L_')} {q[qa[J[n]]]:+.2f}" for n in names if n.startswith('left_')))
  for n in names:
    lo, hi = m.jnt_range[m.joint(n).id]; assert lo - 1e-6 <= q[qa[J[n]]] <= hi + 1e-6, (k, n, q[qa[J[n]]])
  np.save(f'{sys.argv[1]}/v6_{k}.npy', q)
