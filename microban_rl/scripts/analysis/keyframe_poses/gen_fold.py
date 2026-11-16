import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
pp = np.load(f'{sys.argv[1]}/v6_pike_push.npy')
w, x_, y_, z_ = pp[3:7]
xp = np.array([2 * math.atan2(y_, w)] + [pp[qa[J['left_' + k]]] for k in SYM] + [pp[qa[J['left_hip_roll']]], pp[qa[J['left_ankle_roll']]]])
for mg in (0.015, 0.020, 0.030):
  r = solve(lambda f: 0, lambda f: [1e3 * f['hand'] - 5.0, 1e2 * (f['toe'] - mg - f['com']), 1e2 * (f['com'] - f['heel'] - 0.01)], seeds=[xp], n=0)
  # minimise joint change from pike_push subject to the constraints
  def con(x):
    f = feats(build(x)); return np.array([1e3 * (0.0005 - f['flat']), 1e3 * (0.001 - f['foot']), 1e3 * f['other'], 1e3 * f['hand'] - 5.0, 1e2 * (f['toe'] - mg - f['com'])])
  rr = minimize(lambda x: ((x - xp) ** 2).sum(), xp, method='SLSQP', bounds=bounds, constraints=[{'type': 'ineq', 'fun': con}], options=dict(maxiter=500))
  q = build(rr.x); q[0], q[1] = 0.34, 0.0; f = feats(q); tau, til, qe, drift = ST.hold(q); j = int(np.argmax(abs(tau)))
  print(f"margin {mg*1e3:.0f} mm ok={con(rr.x).min() > -0.3}: tilt {ST.tilt(q):5.1f} CoM-toe {(f['com']-f['toe'])*1e3:+.1f} hand z {f['hand']*1e3:.1f} hold {abs(tau).max():.3f} Nm ({names[j]}) tilt after {til:.1f} drift {drift:.3f}")
  print('     ' + ', '.join(f"{n.replace('left_','L_')} {q[qa[J[n]]]:+.2f}" for n in names if n.startswith('left_')) + f", root pitch {rr.x[0]:+.2f}")
  np.save(f'{sys.argv[1]}/v6_fold_{int(mg*1e3)}.npy', q)
