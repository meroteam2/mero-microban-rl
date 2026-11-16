import math
from ik2 import *
from feas import analyse
import sys
nomS=dict(shoulder_pitch=0.6,shoulder_roll=0.35,elbow=0.0,hip_pitch=-0.175,knee=0.0,ankle_pitch=0.0)
res={}
print('== sit, legs straight (heels on floor), hands behind on floor')
for th in (20,30,45,60,70,80):
    pitch=math.radians(th-90)
    q,left,err=solve2(pitch,dict(nomS,hip_pitch=-math.radians(th)),[BUTT,LHAND,LHEEL])
    print(f'th={th} err_mm={np.round(err,1)} joints={ {k:round(float(v),2) for k,v in left.items()} } hand={np.round(site(fk(q),"left_hand"),3)}')
    analyse(f'sit {th} legs straight',q)
print('== sit, legs straight, NO hands')
for th in (20,45,70):
    pitch=math.radians(th-90)
    q,left,err=solve2(pitch,dict(nomS,shoulder_pitch=0.0,shoulder_roll=0.175,elbow=-0.35,hip_pitch=-math.radians(th)),[BUTT,LHEEL],fix=dict(shoulder_pitch=0.0,shoulder_roll=0.175,elbow=-0.35))
    print(f'th={th} err_mm={np.round(err,1)}')
    analyse(f'sit {th} no hands',q)
