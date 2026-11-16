import math, ik, json, sys
from support import *
HARD=lambda n: tuple(m.jnt_range[m.joint(n).id]); ik.SOFT=HARD
import ik2; ik2.SOFT=HARD
from ik2 import solve2
import feas; feas.SOFT=HARD
from feas import analyse
def hipx(d): return d.xpos[m.body('femur__configuration_left').id][0]
XF=float(sys.argv[1]) if len(sys.argv)>1 else 0.13      # foot sole centre x (seat hip at x=0)
inits=[dict(shoulder_pitch=sp,shoulder_roll=0.3,elbow=0.0,hip_pitch=h,knee=k,ankle_pitch=a) for sp,h,k,a in ((0.8,-1.5,0.9,-0.4),(1.2,-1.3,1.6,-0.8),(0.5,-1.0,2.0,-1.0),(1.5,-1.5,2.2,-1.2))]
rows=[]
LEGO=[g for g in COLL if body_group(m.geom_bodyid[g])=='leg(other)' and (m.body(m.geom_bodyid[g]).name.endswith('_2') or 'left' in m.body(m.geom_bodyid[g]).name)]
for hx in [0.0,0.02,0.04,0.06,0.08,0.10,0.12]:
    hz=0.07
    best=None
    for th in range(40,125,5):
        pitch=math.radians(th-90)
        for nom in inits:
            ex=lambda d:[30*(site(d,'left_foot')[0]-XF),100*min(gdist(d,LEGO)-0.002,0)]+list(3*d.xmat[m.body('foot_2').id].reshape(3,3)[:2,2])+[100*min(gdist(d,BUTT)-0.003,0)]
            q,left,err=solve2(pitch,nom,[LFOOT,LHAND],x=hx,z0=hz,extra=ex,w_reg=0.01)
            d=fk(q)
            ok=(max(abs(e) for e in err)<1.0 and abs(site(d,'left_foot')[0]-XF)<0.004 and gdist(d,LEGO)>0.001 and
                np.linalg.norm(d.xmat[m.body('foot_2').id].reshape(3,3)[:2,2])<0.09 and min(mujoco.mj_geomDistance(m,d,g,FLOOR,0.05,np.zeros(6)) for g in COLL)>-0.0015 and gdist(d,BUTT)>0.002)
            if not ok: continue
            sup,cx=support(fk(q - np.eye(m.nq)[2]*0.0003))
            heel=min(p[0] for g in LFOOT for p in [d.geom_xpos[g]]); hand_x=site(d,'left_hand')[0]
            fmin=max(0.0,(heel-cx)/(heel-hand_x)) if heel>hand_x else float('nan')
            cand=(fmin,th,q,left,cx,heel,hand_x)
            if best is None or cand[0]<best[0]: best=cand
    if best is None:
        print(f'hip x={hx:.2f} z={q[2]:.3f}: NO feasible pose (feet flat at x={XF}, hands on floor, butt off)'); continue
    fmin,th,q,left,cx,heel,hand_x=best
    print(f'hip x={hx:.2f} z={q[2]:.3f}: trunk {th}deg, COMx {cx:.3f} heel {heel:.3f} hand {hand_x:.3f} -> static min hand load {fmin:.2f} W (both hands)  joints { {k:round(float(v),2) for k,v in left.items()} }')
    out,_=analyse(f'push hip x={hx}',q,show=False)
    print('    rigid:',json.dumps(out['stiff'],ensure_ascii=False)); print('    realPD:',json.dumps(out['real'],ensure_ascii=False))
    np.save(f'push_{XF:.2f}_{hx:.2f}.npy',q)
