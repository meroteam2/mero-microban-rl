import math
from ik2 import *
def support(d):
    """contact points (x) per group and COM x"""
    pts={}
    for i in range(d.ncon):
        c=d.contact[i]
        if FLOOR not in (c.geom1,c.geom2) or c.dist>1e-3: continue
        g=c.geom2 if c.geom1==FLOOR else c.geom1
        bn=m.body(m.geom_bodyid[g]).name; grp=body_group(m.geom_bodyid[g])
        pts.setdefault(grp,[]).append(c.pos[0])
    return {k:(round(min(v),3),round(max(v),3)) for k,v in pts.items()}, round(float(d.subtree_com[1][0]),3)
def trunk_deg(q): return 90-math.degrees(math.acos(np.clip(R.from_quat(q[[4,5,6,3]]).apply([0,0,1])[2],-1,1)))
