import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
mass = m.body_subtreemass[1]
# Squat family: knee >= 1.2
bsq = list(bounds); bsq[4] = (1.2, bounds[4][1])
bounds[:] = bsq
r = solve(lambda f: (f['com'] - f['toe']) * 1e2, lambda f: [1.0 - 1e3 * f['hand']], seeds=[[0.76, -0.76, 0.0, -1.52, 2.28, -1.52, 0.1, -0.09], [1.2, -1.6, 0.0, -1.5, 2.2, -1.4, 0.1, -0.09], [1.0, -2.2, 0.0, -1.5, 2.0, -1.0, 0.2, 0.0]])
q = build(r.x); f = feats(q); e = f['com'] - f['toe']; h = f['comz']
print(f"SQUAT family, hands on floor: CoM at best {e*1e3:+.1f} mm ahead of toes, height {h:.3f}, tilt {ST.tilt(q):.0f}, hands {(f['hx']-f['toe'])*1e3:.0f} mm ahead | {desc(r.x)}")
v = math.sqrt(9.81 / h) * max(e, 0); print(f"   backward CoM speed needed at hand release {v:.2f} m/s -> mean push {mass*v/0.1:.1f} N (0.1 s) / {mass*v/0.2:.1f} N (0.2 s)")
tau, til, qe, drift = ST.hold(q); print(f"   static hold: torque max {abs(tau).max():.3f} Nm ({abs(tau).max()/ST.TL*100:.0f}%): " + ', '.join(f'{names[j]} {tau[j]:+.3f}' for j in np.argsort(-abs(tau))[:3]))
np.save(f'{sys.argv[1]}/B_squat.npy', q)
