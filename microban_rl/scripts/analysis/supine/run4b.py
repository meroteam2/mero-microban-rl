import math, ik
from support import *
HARD=lambda n: tuple(m.jnt_range[m.joint(n).id]); ik.SOFT=HARD
import ik2; ik2.SOFT=HARD
from ik2 import solve2
def hipx(d): return d.xpos[m.body('femur__configuration_left').id][0]
inits=[dict(shoulder_pitch=0.6,shoulder_roll=0.3,elbow=0.0,hip_pitch=h,knee=k,ankle_pitch=a) for h,k,a in ((-1.4,1.8,-0.6),(-1.2,1.0,-0.4),(-1.5,2.2,-0.9),(-1.0,0.6,-0.2))]
print('min reachable foot distance (sole centre ahead of hip-pitch axis), feet flat + butt touching, hard joint limits')
for th in (55,60,65,70,75,80,90):
    pitch=math.radians(th-90); best=None
    for dx in np.arange(0.14,0.0,-0.01):
        found=None
        for nom in inits:
            ex=lambda d:[30*(site(d,'left_foot')[0]-hipx(d)-dx)]+list(3*d.xmat[m.body('foot_2').id].reshape(3,3)[:2,2])
            q,left,err=solve2(pitch,nom,[BUTT,LFOOT],extra=ex,w_reg=0.01)
            d=fk(q); got=site(d,'left_foot')[0]-hipx(d); tilt=np.degrees(np.arcsin(np.linalg.norm(d.xmat[m.body('foot_2').id].reshape(3,3)[:2,2])))
            # penetration check
            pen=min(mujoco.mj_geomDistance(m,d,g,FLOOR,0.05,np.zeros(6)) for g in COLL)
            if max(abs(e) for e in err)<1.0 and abs(got-dx)<0.004 and tilt<5 and pen>-0.0015:
                found=(dx,left,q); break
        if found: best=found; print(f"   th={th} dx={dx:.2f} ok")
        else: pass
    if best:
        dx,left,q=best; d=fk(q)
        print(f'th={th}: min dx={dx:.2f} m  hip={left["hip_pitch"]:.2f} knee={left["knee"]:.2f} ankle={left["ankle_pitch"]:.2f}  hip z={hipx(d)*0+d.xpos[m.body("femur__configuration_left").id][2]:.3f}')
    else: print(f'th={th}: none even at 0.16')
