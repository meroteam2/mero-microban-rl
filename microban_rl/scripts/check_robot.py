"""Load Microban through MJLab and hold the default pose on CPU MuJoCo for 3 s."""

import mujoco
import numpy as np

from mjlab.entity import Entity
from microban_rl.robot import DEFAULT_JOINT_POS, INIT_HEIGHT, get_microban_robot_cfg

robot = Entity(get_microban_robot_cfg())
spec = robot.spec
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
model = spec.compile()
data = mujoco.MjData(model)
print(f"nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom}")
print("actuators:", [model.actuator(i).name for i in range(model.nu)][:4], "...")
print("kp:", model.actuator_gainprm[0, 0], "forcerange:", model.actuator_forcerange[0])

data.qpos[2] = INIT_HEIGHT
for name, q in DEFAULT_JOINT_POS.items():
  data.qpos[model.joint(name).qposadr[0]] = q
for i in range(model.nu):
  data.ctrl[i] = DEFAULT_JOINT_POS[model.joint(model.actuator_trnid[i, 0]).name]
mujoco.mj_forward(model, data)
print("initial contacts:", data.ncon)
for t in range(int(3.0 / model.opt.timestep)):
  mujoco.mj_step(model, data)
  if t % 500 == 0:
    up = data.xmat[1].reshape(3, 3)[2, 2]
    print(f"t={t * model.opt.timestep:.1f}s z={data.qpos[2]:.4f} upright={up:.3f} ncon={data.ncon}")
print("final trunk z:", round(float(data.qpos[2]), 4))
print("contacts:", sorted({(model.geom(c.geom1).name or model.body(model.geom_bodyid[c.geom1]).name,
                            model.geom(c.geom2).name or model.body(model.geom_bodyid[c.geom2]).name)
                           for c in data.contact[:data.ncon]}))
