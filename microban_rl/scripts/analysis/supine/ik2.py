from ik import *
from scipy.optimize import least_squares
G=lambda *names: [m.geom(n).id if isinstance(n,str) else n for n in names]
BUTT=[g for g in COLL if m.body(m.geom_bodyid[g]).name in ('trunk','hip_block__configuration_left','hip_block__configuration_right')]
LHAND=[g for g in COLL if m.geom_bodyid[g]==m.body('radius_2').id]
LHAND_TIP=[64]
LFOOT=[g for g in COLL if m.geom_bodyid[g]==m.body('foot_2').id]
LHEEL=[m.geom('left_foot_collision_3').id, m.geom('left_foot_collision_4').id]
_ft=np.zeros(6)
def gdist(d,gs): return min(mujoco.mj_geomDistance(m,d,g,FLOOR,1.0,_ft) for g in gs)
def solve2(pitch, nominal, touch, x=0.0, z0=0.05, w_reg=0.05, extra=None, fix=None):
    """touch: list of geom-groups that must touch the floor; everything else must stay above."""
    fix=fix or {}
    names=[n for n in VARS if n not in fix]
    lo=[SOFT('left_'+n)[0] for n in names]+[-0.05]; hi=[SOFT('left_'+n)[1] for n in names]+[0.3]
    x0=np.clip(np.array([nominal[n] for n in names]+[z0]),np.array(lo)+1e-6,np.array(hi)-1e-6)
    touched=set(g for grp in touch for g in grp)
    def unpack(v): left=dict(nominal); left.update(fix); left.update(dict(zip(names,v[:-1]))); return left,v[-1]
    def res(v):
        left,z=unpack(v); d=fk(make_q(x,z,pitch,left)); r=[]
        for grp in touch: r.append(100*gdist(d,grp))
        for g in COLL:
            if g in touched: continue
            if m.body(m.geom_bodyid[g]).name.endswith('_2') or 'left' in m.body(m.geom_bodyid[g]).name or m.geom_bodyid[g] in (1,2):
                dd=mujoco.mj_geomDistance(m,d,g,FLOOR,0.05,_ft); r.append(100*min(dd,0))
        if extra: r+=list(extra(d))
        r+=[w_reg*(v[i]-nominal[n]) for i,n in enumerate(names)]
        return np.array(r)
    sol=least_squares(res,x0,bounds=(lo,hi))
    left,z=unpack(sol.x); q=make_q(x,z,pitch,left); d=fk(q)
    errs=[gdist(d,grp)*1e3 for grp in touch]
    return q,left,errs
