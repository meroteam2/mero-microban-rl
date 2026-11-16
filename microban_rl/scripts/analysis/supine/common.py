import mujoco, numpy as np, joblib
from scipy.spatial.transform import Rotation as R
XML='/home/mero/mero/keyframe_editor/microban_scene.xml'
m=mujoco.MjModel.from_xml_path(XML)
KP,KD,TAU=0.88,0.01,1.75*0.366
DEFAULT={"head":0.0,"right_shoulder_pitch":0.0,"right_shoulder_roll":-0.175,"right_elbow":-0.349,"right_hip_yaw":0.0,"right_hip_roll":-0.087,"right_hip_pitch":-0.175,"right_knee":0.0,"right_ankle_pitch":0.0,"right_ankle_roll":0.087,"left_shoulder_pitch":0.0,"left_shoulder_roll":0.175,"left_elbow":-0.349,"left_hip_yaw":0.0,"left_hip_roll":0.087,"left_hip_pitch":-0.175,"left_knee":0.0,"left_ankle_pitch":0.0,"left_ankle_roll":-0.087}
ACT=[m.actuator(i).name for i in range(m.nu)]
QADR=np.array([m.jnt_qposadr[m.joint(n).id] for n in ACT]); VADR=np.array([m.jnt_dofadr[m.joint(n).id] for n in ACT])
RANGE=np.array([m.jnt_range[m.joint(n).id] for n in ACT])
FLOOR=m.geom('floor').id
COLL=[g for g in range(m.ngeom) if g!=FLOOR and (m.geom_contype[g] or m.geom_conaffinity[g])]
def body_group(b):
    n=m.body(b).name
    if n in ('radius','radius_2','humerus__configuration_right','humerus__configuration_left','shoulder','shoulder_2'): return 'arm'
    if n in ('foot','foot_2'): return 'foot'
    if n in ('trunk','head','hip','hip_2','hip_block__configuration_right','hip_block__configuration_left'): return 'trunk/pelvis'
    return 'leg(other)'
def qpos_from(pos,quat,joints):
    q=np.zeros(m.nq); q[:3]=pos; q[3:7]=quat
    for n,v in joints.items(): q[m.jnt_qposadr[m.joint(n).id]]=v
    return q
def min_floor_dist(d):
    ft=np.zeros(6); return min(mujoco.mj_geomDistance(m,d,g,FLOOR,1.0,ft) for g in COLL)
def drop_to_floor(q,clear=0.0):
    d=mujoco.MjData(m); d.qpos[:]=q; mujoco.mj_forward(m,d)
    q=q.copy(); q[2]-=min_floor_dist(d)-clear; return q
def sim(q, target=None, secs=1.5, kp=KP, kd=KD, tau=TAU, record=0.3):
    """PD hold; returns data and averages over last `record` s of actuator torque and contact forces by group."""
    d=mujoco.MjData(m); d.qpos[:]=q
    tgt=q[QADR].copy() if target is None else target
    n=int(secs/m.opt.timestep); nr=int(record/m.opt.timestep)
    taus=[]; forces={}; com=[]
    for k in range(n):
        t=kp*(tgt-d.qpos[QADR])-kd*d.qvel[VADR]
        d.ctrl[:]=np.clip(t,-tau,tau) if tau else t
        mujoco.mj_step(m,d)
        if k>=n-nr:
            taus.append(d.ctrl.copy()); com.append(d.subtree_com[1].copy())
            f=contact_forces(d)
            for g,v in f.items(): forces[g]=forces.get(g,0)+v/nr
    return d, np.mean(taus,0), forces, np.mean(com,0)
def contact_forces(d):
    out={}; f6=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i]
        if FLOOR not in (c.geom1,c.geom2): continue
        g=c.geom2 if c.geom1==FLOOR else c.geom1
        mujoco.mj_contactForce(m,d,i,f6)
        fw=c.frame.reshape(3,3).T@f6[:3]   # world frame force (normal first row)
        if c.geom1!=FLOOR: fw=-fw
        side=''
        bn=m.body(m.geom_bodyid[g]).name
        grp=body_group(m.geom_bodyid[g])
        if grp in('arm','foot','leg(other)'): side='L ' if (bn.endswith('_2') or 'left' in bn) else 'R '
        if grp=='foot': side='L ' if bn=='foot_2' else 'R '
        key=side+grp
        out[key]=out.get(key,np.zeros(3))+np.abs(fw)*np.array([1,1,1])*0+fw
    return out
W=m.body_subtreemass[1]*9.81
