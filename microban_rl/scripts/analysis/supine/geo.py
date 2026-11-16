import mujoco, numpy as np, joblib
m=mujoco.MjModel.from_xml_path('/home/mero/mero/keyframe_editor/microban_scene.xml'); d=mujoco.MjData(m)
v6=joblib.load('/home/mero/mero/keyframe_editor/keyframes/getup_v6.lz4')
kf={k['name']:k for k in v6['keyframes']}
def show(q,label):
    d.qpos[:]=q; mujoco.mj_forward(m,d)
    s=lambda n: np.round(d.site_xpos[m.site(n).id],3)
    print(label,'root',np.round(q[:3],3),'COM',np.round(d.subtree_com[1],3),'Lfoot',s('left_foot'),'Lhand',s('left_hand'))
    # foot sole extents x
    fx=[d.geom_xpos[g][0] for g in range(m.ngeom) if m.geom(g).name.startswith('left_foot_collision')]
    print('   foot sole x range', np.round(min(fx),3), np.round(max(fx),3))
show(kf['crouch']['qpos'],'crouch'); show(kf['stand_target']['qpos'],'stand')
q=m.key('home').qpos.copy()
a=lambda n: m.jnt_qposadr[m.joint(n).id]
for j,v in [('left_shoulder_pitch',0.8),('left_shoulder_pitch',-0.8),('left_elbow',-1.2),('left_elbow',0.5),('left_hip_pitch',-0.8),('left_knee',1.0),('left_ankle_pitch',-0.5)]:
    qq=q.copy(); qq[a(j)]=v; d.qpos[:]=qq; mujoco.mj_forward(m,d)
    print(j,v,'hand',np.round(d.site_xpos[m.site('left_hand').id],3),'foot',np.round(d.site_xpos[m.site('left_foot').id],3), 'knee', np.round(d.xpos[m.body('tibia__configuration_left').id],3))
