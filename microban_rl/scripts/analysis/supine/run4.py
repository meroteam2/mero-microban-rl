import math, ik
from support import *
# use HARD limits for feasibility
HARD=lambda n: tuple(m.jnt_range[m.joint(n).id])
ik.SOFT=HARD
import ik2; ik2.SOFT=HARD
from ik2 import solve2
nom=dict(shoulder_pitch=0.6,shoulder_roll=0.3,elbow=0.0,hip_pitch=-1.4,knee=1.8,ankle_pitch=-0.6)
print('foot sole centre distance in front of hip joint (dx) reachable with feet flat, butt on floor:')
for th in (60,70,80,90):
    pitch=math.radians(th-90); row=[]
    for dx in (0.14,0.12,0.10,0.08,0.06,0.04):
        q,left,err=solve2(pitch,nom,[BUTT,LFOOT],extra=lambda d:[30*(site(d,'left_foot')[0]-d.xpos[m.body('femur__configuration_left').id][0]-dx), 5*fk_flat(d)] if False else
                          [30*(site(d,'left_foot')[0]-d.xpos[m.body('femur__configuration_left').id][0]-dx)]+list(2*d.xmat[m.body('foot_2').id].reshape(3,3)[:2,2]))
        d=fk(q); got=site(d,'left_foot')[0]-d.xpos[m.body('femur__configuration_left').id][0]
        ok=max(abs(e) for e in err)<1.5 and abs(got-dx)<0.005
        row.append(f"dx={dx:.2f}:{'OK ' if ok else 'NO '}hip={left['hip_pitch']:.2f} knee={left['knee']:.2f} ank={left['ankle_pitch']:.2f}(got {got:.3f})")
    print(f'th={th}:', ' | '.join(row))
