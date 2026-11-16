from common import *
from scipy.optimize import least_squares
MIRROR={'shoulder_pitch':1,'shoulder_roll':-1,'elbow':1,'hip_yaw':-1,'hip_roll':-1,'hip_pitch':1,'knee':1,'ankle_pitch':1,'ankle_roll':-1}
VARS=['shoulder_pitch','shoulder_roll','elbow','hip_pitch','knee','ankle_pitch']
SOFT=lambda n: (lambda r:(r.mean()-0.45*(r[1]-r[0]), r.mean()+0.45*(r[1]-r[0])))(m.jnt_range[m.joint(n).id])
def joints_from(left):
    j=dict(DEFAULT)
    for k,v in left.items():
        j['left_'+k]=v; j['right_'+k]=MIRROR[k]*v
    return j
def make_q(x,z,pitch,left):
    quat=R.from_euler('y',pitch).as_quat(); return qpos_from((x,0,z),(quat[3],*quat[:3]),joints_from(left))
d_=mujoco.MjData(m)
def fk(q):
    d_.qpos[:]=q; mujoco.mj_forward(m,d_); return d_
def site(d,n): return d.site_xpos[m.site(n).id].copy()
def sole_pts(d,side='left'):
    gs=[m.geom(f'{side}_foot_collision_{i}').id for i in range(1,7)]; return np.array([d.geom_xpos[g] for g in gs])
def solve(pitch, nominal, free_root=('x','z'), root0=(0,0.05), hand=None, foot=None, foot_flat=True, extra=None, w_reg=0.05, fixed_root=None):
    """hand/foot: dict of target coords {'x':..,'z':..} for left_hand / left_foot site."""
    names=VARS
    lo=[SOFT('left_'+n)[0] for n in names]+[-1,-0.1]; hi=[SOFT('left_'+n)[1] for n in names]+[1,0.3]
    x0=np.array([nominal[n] for n in names]+list(root0)); x0=np.clip(x0,np.array(lo)+1e-6,np.array(hi)-1e-6)
    def res(v):
        left=dict(zip(names,v[:6])); x,z=v[6],v[7]
        if fixed_root is not None:
            x=fixed_root.get('x',x); z=fixed_root.get('z',z)
        d=fk(make_q(x,z,pitch,left)); r=[]
        if hand:
            h=site(d,'left_hand')
            for k,t in hand.items(): r.append(20*(h['xyz'.index(k)]-t))
        if foot:
            f=site(d,'left_foot')
            for k,t in foot.items(): r.append(20*(f['xyz'.index(k)]-t))
            if foot_flat:
                fz=d.xmat[m.body('foot_2').id].reshape(3,3)[:,2]; r+= [2*fz[0],2*fz[1]]
        if extra: r+=list(extra(d))
        r+=[w_reg*(v[i]-nominal[n]) for i,n in enumerate(names)]
        return np.array(r)
    sol=least_squares(res,x0,bounds=(lo,hi))
    v=sol.x; left=dict(zip(names,v[:6])); x,z=v[6],v[7]
    if fixed_root is not None: x=fixed_root.get('x',x); z=fixed_root.get('z',z)
    return make_q(x,z,pitch,left), left, sol
