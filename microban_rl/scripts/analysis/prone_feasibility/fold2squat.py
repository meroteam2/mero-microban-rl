import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
# knee from straight (-0.76) to squat (1.73); solve hip, ankle, root pitch so CoM over sole middle (+-), soles flat; arms hanging forward (sh -1.7)
print(' knee    hip   ankle  tilt  CoM-mid(mm) hold Nm (joint)')
prev = [-1.54, 0.4, 1.8]
for knee in np.linspace(-0.76, 1.73, 11):
  lo = [m.jnt_range[m.joint('left_hip_pitch').id], m.jnt_range[m.joint('left_ankle_pitch').id]]
  def P(y): return build(np.array([y[2], -1.7, -0.07, np.clip(y[0], lo[0][0], 0), knee, np.clip(y[1], *lo[1]), 0.15, 0.0]))
  def obj(y):
    f = feats(P(y)); mid = (f['heel'] + f['toe']) / 2
    pen = 1e3 * max(0, -f['other']) ** 1
    return 1e4 * (f['com'] - mid) ** 2 + 1e5 * f['flat'] ** 2 + pen + 1e3 * max(0, f['foot'] - 0.001)
  best = None
  for y0 in (prev, [-1.54, -0.5, 1.2], [-1.2, -1.0, 0.8]):
    r = minimize(obj, y0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-4, fatol=1e-9))
    if best is None or r.fun < best.fun: best = r
  prev = list(best.x); q = P(best.x); f = feats(q)
  tau, til, qe, drift = ST.hold(q); j = int(np.argmax(abs(tau)))
  print(f"{knee:+.2f}  {np.clip(best.x[0], lo[0][0], 0):+.2f}  {np.clip(best.x[1], *lo[1]):+.2f}  {ST.tilt(q):5.1f}  {(f['com']-(f['heel']+f['toe'])/2)*1e3:+6.1f}   flat {f['flat']*1e3:.1f}  other {f['other']*1e3:+.1f}mm  {abs(tau).max():.3f} ({names[j]}) tilt-after {til:.0f}")
