from ik import *
import math, json
ARM=[n for n in ACT if 'shoulder' in n or 'elbow' in n]
def analyse(label,q,stiff_secs=1.0, show=True):
    q=drop_to_floor(q,0.0005)
    d0=fk(q); com0=d0.subtree_com[1].copy()
    # joint range check
    jl=[]
    for n in ACT:
        v=q[m.jnt_qposadr[m.joint(n).id]]; lo,hi=m.jnt_range[m.joint(n).id]; slo,shi=SOFT(n)
        if v<lo-1e-6 or v>hi+1e-6: jl.append(f'{n}={v:.2f} OUT[{lo:.2f},{hi:.2f}]')
        elif v<slo or v>shi: jl.append(f'{n}={v:.2f} >soft')
    # stiff hold: static torque & contact distribution
    ds,tau_s,f_s,com_s=sim(q,secs=stiff_secs,kp=30.0,kd=0.3,tau=None)
    drift_s=np.linalg.norm(ds.qpos[:3]-q[:3]); rot_s=(R.from_quat(ds.qpos[[4,5,6,3]]).inv()*R.from_quat(q[[4,5,6,3]])).magnitude()
    # real PD hold
    dr,tau_r,f_r,com_r=sim(q,secs=1.5)
    drift_r=np.linalg.norm(dr.qpos[:3]-q[:3]); rot_r=(R.from_quat(dr.qpos[[4,5,6,3]]).inv()*R.from_quat(q[[4,5,6,3]])).magnitude()
    sag=np.abs(dr.qpos[QADR]-q[QADR]).max(); sagj=ACT[int(np.abs(dr.qpos[QADR]-q[QADR]).argmax())]
    top=np.argsort(-np.abs(tau_s))[:4]
    trunk_ang=90-math.degrees(math.acos(np.clip(R.from_quat(q[[4,5,6,3]]).apply([0,0,1])[2],-1,1)))
    out=dict(label=label,trunk_deg=round(trunk_ang,1),com=np.round(com0,3).tolist(),
             stiff=dict(drift_mm=round(drift_s*1e3,1),rot_deg=round(math.degrees(rot_s),1),tau={ACT[i]:round(float(tau_s[i]),3) for i in top},
                        load={k:round(float(v[2]/W),2) for k,v in sorted(f_s.items())},fric={k:round(float(np.hypot(v[0],v[1])/max(v[2],1e-9)),2) for k,v in sorted(f_s.items()) if v[2]>0.02*W}),
             real=dict(drift_mm=round(drift_r*1e3,1),rot_deg=round(math.degrees(rot_r),1),max_sag=f'{sag:.2f} ({sagj})',max_tau=round(float(np.abs(tau_r).max()),3),
                       saturated=[ACT[i] for i in range(m.nu) if abs(tau_r[i])>0.98*TAU],load={k:round(float(v[2]/W),2) for k,v in sorted(f_r.items())}),
             limits=jl)
    if show: print(json.dumps(out,ensure_ascii=False))
    return out, ds
