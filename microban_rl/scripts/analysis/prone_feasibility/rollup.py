"""Quasi-static roll-up from the forward fold (pike, CoM over feet) to stand_target: for hip_pitch
from -1.54 to -0.175, choose knee/ankle so the CoM stays over the sole centre and soles stay flat;
holding torque at each step (stiff PD, feet only)."""
import sys
exec(open(f'{sys.argv[1]}/feasibility.py').read().split('# A)')[0])
names_ = names
def pose(hip, knee, ankle, pitch, hr=0.0, ar=0.0, sh=-0.5):
  return build(np.array([pitch, sh, -0.35, hip, knee, ankle, hr, ar]))
prev = None
print(' hip   knee  ankle  tilt  CoM-mid(mm)  hold torque max (joint)')
for hip in np.linspace(-1.54, -0.175, 12):
  def obj(y):
    f = feats(pose(hip, y[0], y[1], y[2], 0.087, -0.087)); mid = (f['heel'] + f['toe']) / 2
    return 1e4 * (f['com'] - mid) ** 2 + 1e5 * f['flat'] ** 2 + 0.01 * (y[0] ** 2)
  best = None
  for y0 in ([0.0, 0.0, 1.5], [0.8, -0.6, 1.0], [-0.5, 0.3, 1.8], [1.5, -1.0, 0.8]) if prev is None else ([*prev],):
    r = minimize(obj, y0, method='Nelder-Mead', options=dict(maxiter=2000, xatol=1e-4, fatol=1e-8))
    lo_k, hi_k = m.jnt_range[m.joint('left_knee').id]; lo_a, hi_a = m.jnt_range[m.joint('left_ankle_pitch').id]
    if lo_k <= r.x[0] <= hi_k and lo_a <= r.x[1] <= hi_a and (best is None or r.fun < best.fun): best = r
  if best is None: print(f'{hip:+.2f}  no in-range solution'); prev = None; continue
  prev = best.x; q = pose(hip, *best.x, 0.087, -0.087); f = feats(q)
  tau, til, qe, drift = ST.hold(q); j = int(np.argmax(abs(tau)))
  print(f"{hip:+.2f} {best.x[0]:+.2f} {best.x[1]:+.2f}  {ST.tilt(q):5.1f}  {(f['com']-(f['heel']+f['toe'])/2)*1e3:+6.1f}  flat {f['flat']*1e3:.1f}mm   {abs(tau).max():.3f} Nm ({names_[j]}) drift {drift:.3f}")
