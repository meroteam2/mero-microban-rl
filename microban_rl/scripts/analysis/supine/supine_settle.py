from common import *
import math
Q=(math.cos(-math.pi/4),0,math.sin(-math.pi/4),0)
q=qpos_from((0,0,0.2),Q,DEFAULT); q0=drop_to_floor(q)
print('touch height', q0[2])
d,tau,f,com=sim(drop_to_floor(q,0.01),secs=2.0)
rot=R.from_quat(d.qpos[[4,5,6,3]]); bx=rot.apply([1,0,0]); bz=rot.apply([0,0,1])
print('settled z %.4f belly dir %s head dir %s |qvel|max %.4f'%(d.qpos[2],np.round(bx,3),np.round(bz,3),np.abs(d.qvel).max()))
print('joint dev from target', np.round(d.qpos[QADR]-q[QADR],3).tolist())
print('tau', dict(zip(ACT,np.round(tau,3))))
print('contacts', {k:np.round(v/W,3) for k,v in f.items()}, 'COM',np.round(com,3))
print('euler xyz', R.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz'))
np.save('supine_settled.npy', d.qpos.copy())
# randomized spawns
rng=np.random.default_rng(0); lo=RANGE.mean(1)-0.45*(RANGE[:,1]-RANGE[:,0]); hi=RANGE.mean(1)+0.45*(RANGE[:,1]-RANGE[:,0])
for H in (0.05,0.055,0.06):
  pen=[];bad=0;zs=[];bxz=[]
  for i in range(150):
    yaw,roll,pitch=rng.uniform(-np.pi,np.pi),*rng.uniform(-0.15,0.15,2)
    qr=R.from_euler('z',yaw)*R.from_euler('xyz',[roll,pitch,0])*R.from_quat([Q[1],Q[2],Q[3],Q[0]])
    xq=qr.as_quat(); qq=q.copy(); qq[2]=H; qq[3:7]=[xq[3],*xq[:3]]
    qq[QADR]=np.clip(q[QADR]+rng.uniform(-0.15,0.15,m.nu),lo,hi)
    dd=mujoco.MjData(m); dd.qpos[:]=qq; mujoco.mj_forward(m,dd); pen.append(min_floor_dist(dd))
    dd,_,_,_=sim(qq,secs=1.0,record=0.05)
    b=R.from_quat(dd.qpos[[4,5,6,3]]).apply([1,0,0])[2]; bxz.append(b); zs.append(dd.qpos[2])
  pen=np.array(pen)
  print(f'H={H}: spawn min dist  min {pen.min()*1e3:.1f} mm, p5 {np.percentile(pen,5)*1e3:.1f}, max {pen.max()*1e3:.1f}; penetrating {np.sum(pen<0)}/150; settled z {np.min(zs):.4f}-{np.max(zs):.4f}; belly-up z-comp min {np.min(bxz):.3f}')
