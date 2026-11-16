from feas import *
sup=np.load('supine_settled.npy')
analyse('P1 supine (settled)',sup)
# sit-up path: legs straight, hands on floor behind hips
nom=dict(shoulder_pitch=0.5,shoulder_roll=0.3,elbow=0.0,hip_pitch=-0.175,knee=0.0,ankle_pitch=0.0)
for th in (15,30,45,60,70,80):
    pitch=math.radians(th-90)
    def ex(d):  # heels on floor: lowest sole point z=0 ; butt on floor handled by drop
        return []
    # hip flexion so legs lie on floor: thigh horizontal -> hip_pitch = -(th) roughly ; solve: feet z=0.007 (site), flat not needed
    q,left,sol=solve(pitch,dict(nom,hip_pitch=-math.radians(th)),root0=(0,0.03),hand={'z':0.0055},foot={'z':0.007},foot_flat=False,
                     extra=lambda d:[20*(d.xpos[m.body('hip_block__configuration_left').id][2]-0.012)])
    print(f'th={th} left={ {k:round(v,2) for k,v in left.items()} } hand={np.round(site(fk(q),"left_hand"),3)} hip={np.round(q[:3],3)}')
    analyse(f'P3 sit {th}deg legs straight, hands back',q)
