import math, json
from support import *
from feas import analyse
v6=joblib.load('/home/mero/mero/keyframe_editor/keyframes/getup_v6.lz4'); kf={k['name']:k for k in v6['keyframes']}
for n in ('squat_deep','crouch','stand_target'):
    q=kf[n]['qpos'].astype(float); d=fk(q)
    sup,cx=support(fk(drop_to_floor(q,-0.0003)))
    out,_=analyse(n,q,show=False)
    print(n,'trunk_deg(from floor)',out['trunk_deg'],'COMx',cx,'support',sup)
    print('   rigid',json.dumps(out['stiff'])); print('   real ',json.dumps(out['real'])); print('   limits',out['limits'])
M=m.body_subtreemass[1]
print('--- dynamic estimate for the hands-push gap')
for f in ('push_0.13_0.02.npy','push_0.13_0.00.npy','push_0.10_0.00.npy'):
    q=np.load(f); d=fk(q); c=d.subtree_com[1].copy()
    heel=min(d.geom_xpos[g][0] for g in LFOOT)
    # whole-body inertia about COM around y
    Ic=0.0
    for b in range(1,m.nbody):
        rr=d.xipos[b]-c; Rb=d.ximat[b].reshape(3,3); Ib=Rb@np.diag(m.body_inertia[b])@Rb.T
        Ic+=Ib[1,1]+m.body_mass[b]*(rr[0]**2+rr[2]**2)
    dx=heel-c[0]; r=math.hypot(dx,c[2]); dh=r-c[2]
    v=math.sqrt(2*9.81*dh*M*r*r/(M*r*r+Ic))
    J=M*v
    sh=d.xpos[m.body('shoulder_2').id]; hd=site(d,'left_hand')
    print(f'{f}: COM ({c[0]:.3f},{c[2]:.3f}) heel x {heel:.3f}: COM {dx*100:.1f} cm behind heel, pivot radius {r:.3f}, COM must rise {dh*100:.1f} cm; '
          f'Ic {Ic*1e4:.2f} kg cm^2 -> COM speed at hand release >= {v:.2f} m/s, impulse >= {J:.2f} N s; '
          f'shoulder height above hand {sh[2]-hd[2]:.3f} m, behind-hand dx {sh[0]-hd[0]:.3f}')
    for T in (0.1,0.15,0.2):
        print(f'     push over {T:.2f} s: mean extra force {J/T:.1f} N = {J/T/W:.2f} W (on top of static 0.5 W); travel ~{0.5*v*T*100:.1f} cm')
print('W =',round(W,2),'N; one-arm horizontal force cap from shoulder torque 0.64 Nm with lever = shoulder height above hand:')
