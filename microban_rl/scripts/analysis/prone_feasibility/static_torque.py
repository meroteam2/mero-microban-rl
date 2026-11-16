"""Static holding torque per keyframe: place kinematically on the floor, hold with a stiff PD
(no torque limit) for 1.5 s, report steady torques, drift, CoM vs support."""
import math, sys, numpy as np, joblib, mujoco
from microban_rl.robot import XL330_EFFORT_LIMIT as TL
SCENE='/home/mero/mero/keyframe_editor/microban_scene.xml'
m = mujoco.MjModel.from_xml_path(SCENE)
m.actuator_forcelimited[:] = 0; m.actuator_ctrllimited[:] = 0
d = mujoco.MjData(m)
names=[m.actuator(i).name for i in range(m.nu)]
qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in names]); va=np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
floor=m.geom('floor').id
def lowest_by_body(q):
  d.qpos[:]=q; mujoco.mj_forward(m,d); out={}
  for g in range(m.ngeom):
    if g==floor or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
    R=d.geom_xmat[g].reshape(3,3); s=m.geom_size[g]; p=d.geom_xpos[g]; t=m.geom_type[g]
    if t==mujoco.mjtGeom.mjGEOM_BOX: z=p[2]-(abs(R[2])*s).sum()
    elif t==mujoco.mjtGeom.mjGEOM_CYLINDER: cz=abs(R[2,2]); z=p[2]-cz*s[1]-s[0]*np.sqrt(max(0,1-cz*cz))
    else: continue
    b=m.body(m.geom_bodyid[g]).name; out[b]=min(out.get(b,9e9),z)
  return out
def tilt(q):
  w,x,y,z=q[3:7]; R=np.zeros(9); mujoco.mju_quat2Mat(R,np.array([w,x,y,z])); return math.degrees(math.acos(np.clip(R[8],-1,1)))
def support(q):
  """CoM x and x-extent of the touching parts (feet corners, hand tips)."""
  d.qpos[:]=q; mujoco.mj_forward(m,d)
  com=d.subtree_com[1][0]; xs={}
  for i in range(d.ncon):
    c=d.contact[i]; g=c.geom2 if c.geom1==floor else c.geom1
    b=m.body(m.geom_bodyid[g]).name; xs.setdefault(b,[]).append(c.pos[0])
  return com, {b:(min(v),max(v)) for b,v in xs.items()}
def hold(q, kp=30.0, kd=0.3, T=1.5):
  d.qpos[:]=q; d.qvel[:]=0; mujoco.mj_forward(m,d); tgt=q[qa].copy(); hist=[]
  for k in range(int(T/m.opt.timestep)):
    d.ctrl[:]=kp*(tgt-d.qpos[qa])-kd*d.qvel[va]; mujoco.mj_step(m,d)
    if k*m.opt.timestep>T-0.5: hist.append(d.ctrl.copy())
  tau=np.mean(hist,0); drift=np.abs(d.qpos[qa]-tgt).max()
  return tau, tilt(d.qpos), d.qpos.copy(), drift
def place(q):
  q=q.copy(); q[2]-=min(lowest_by_body(q).values()); return q
def report(name, q):
  q=place(q); com,sup=support(q)
  tau,til,qe,drift=hold(q)
  top=np.argsort(-abs(tau))[:3]
  sup_s=' '.join(f"{b}[{lo:.3f},{hi:.3f}]" for b,(lo,hi) in sorted(sup.items()))
  print(f"{name:14s} tilt {tilt(q):5.1f}->{til:5.1f}  CoMx {com:+.3f}  support: {sup_s}")
  print(f"{'':14s} hold torque max {abs(tau).max():.3f} Nm ({abs(tau).max()/TL*100:.0f}% of limit): " + ", ".join(f"{names[j]} {tau[j]:+.3f}" for j in top) + f" | drift {drift:.3f} rad")
  return q
if __name__=='__main__':
  dd=joblib.load(sys.argv[1]); kf={k['name']:k for k in dd['keyframes']}
  for n in sys.argv[2:]:
    q=np.asarray(kf[n]['qpos'],dtype=np.float64).copy(); q[qa]=kf[n]['motor_pos']; report(n,q)
