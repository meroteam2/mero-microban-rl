from support import *
q=np.load('supine_settled.npy'); d=fk(q)
legs=lambda b: any(s in m.body(b).name for s in ('hip','femur','tibia','ankle','foot'))
hipaxis=(d.xanchor[m.joint('left_hip_pitch').id]+d.xanchor[m.joint('right_hip_pitch').id])/2
def mom(sel):
    mm=sum(m.body_mass[b] for b in range(1,m.nbody) if sel(b)); cx=sum(m.body_mass[b]*d.xipos[b][0] for b in range(1,m.nbody) if sel(b))/mm
    return mm,cx-hipaxis[0]
mu,du=mom(lambda b: not legs(b) or m.body(b).name in ('hip','hip_2')); ml,dl=mom(lambda b: legs(b) and m.body(b).name not in ('hip','hip_2'))
print(f'hip-pitch axis x {hipaxis[0]:.3f}; upper body {mu:.3f} kg COM {du*100:.1f} cm from hip axis -> {mu*9.81*abs(du):.3f} Nm; legs (below hip-pitch) {ml:.3f} kg COM {dl*100:.1f} cm -> {ml*9.81*abs(dl):.3f} Nm; hip-flexor max 2 x 0.64 = 1.28 Nm')
