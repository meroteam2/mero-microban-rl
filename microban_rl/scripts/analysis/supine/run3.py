import math
from support import *
from feas import analyse
nomS=dict(shoulder_pitch=0.6,shoulder_roll=0.35,elbow=0.0,hip_pitch=-0.175,knee=0.0,ankle_pitch=0.0)
print('== support geometry of sit poses (rigid hold result)')
for th,hands in [(t,h) for h in (True,False) for t in (35,40,45,50,55,60,65)]:
    pitch=math.radians(th-90)
    if hands: q,left,err=solve2(pitch,dict(nomS,hip_pitch=-math.radians(th)),[BUTT,LHAND,LHEEL])
    else:
        fx=dict(shoulder_pitch=0.0,shoulder_roll=0.175,elbow=-0.35)
        q,left,err=solve2(pitch,dict(nomS,hip_pitch=-math.radians(th),**fx),[BUTT,LHEEL],fix=fx)
    sup,cx=support(fk(drop_to_floor(q,-0.0003))); q=drop_to_floor(q,0.0005)
    ds,_,_,_=sim(q,secs=1.0,kp=30.0,kd=0.3,tau=None); dr,_,_,_=sim(q,secs=1.5)
    print(f'th={th} hands={hands}: start support {sup} COMx {cx}  -> rigid end trunk {trunk_deg(ds.qpos):.0f}deg, real-PD end trunk {trunk_deg(dr.qpos):.0f}deg')
