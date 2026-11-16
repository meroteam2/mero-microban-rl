"""Microban humanoid entity config for MJLab.

The MJCF comes straight from the Rhoban/microban repo (src/model/mjcf/robot.xml).
Its torque <motor> actuators are stripped and replaced by MJLab position
actuators that mimic the Dynamixel XL330 position loop used on the real robot.
"""

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF.
##

# Override with MICROBAN_ROOT=/path/to/microban if the repo lives elsewhere.
MICROBAN_ROOT = Path(
  os.environ.get("MICROBAN_ROOT", Path(__file__).resolve().parents[3] / "microban")
)
MICROBAN_XML = MICROBAN_ROOT / "src" / "model" / "mjcf" / "robot.xml"


def get_spec() -> mujoco.MjSpec:
  if not MICROBAN_XML.exists():
    raise FileNotFoundError(
      f"Microban MJCF not found at {MICROBAN_XML}. "
      "Clone https://github.com/Rhoban/microban next to this project "
      "or set MICROBAN_ROOT."
    )
  spec = mujoco.MjSpec.from_file(str(MICROBAN_XML))
  # Drop the XML torque motors; MJLab adds its own position actuators below.
  for act in list(spec.actuators):
    spec.delete(act)
  return spec


##
# Actuators (Dynamixel XL330-M288-T).
##

# From microban/src/constants.py: KP_DEFAULT = 400 register units * 0.0022 Nm/rad
# (the gain the real robot uses when holding a pose). The softer walking gain
# KP_RL = 125 also works but makes standing much harder to learn.
XL330_STIFFNESS = 400 * 0.0022
XL330_DAMPING = 0.01
# Firmware current limit (1.75 A) * torque constant (0.366 Nm/A).
XL330_EFFORT_LIMIT = 1.75 * 0.366

XL330_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  # Joint-name regex (a bare ".*" would also match the IMU/foot sites).
  target_names_expr=("head", ".*_shoulder_.*", ".*_elbow", ".*_hip_.*", ".*_knee",
                     ".*_ankle_.*"),
  stiffness=XL330_STIFFNESS,
  damping=XL330_DAMPING,
  effort_limit=XL330_EFFORT_LIMIT,
)

# Speed limit. ROBOTIS e-manual, XL330-M288-T: no-load speed 103 rpm @ 5.0 V,
# 123 rpm @ 6.0 V (rated maximum). This model needs no separate velocity limit:
# the MJCF "xl330" joint damping (0.041 Nm s/rad) is the motor's back-EMF
# (kt^2 / R = 0.366^2 / 2.811 = 0.048, microban BAM constants), so with the
# 0.64 Nm torque limit an unloaded joint saturates at
# (0.64 - 0.013 friction) / (0.041 + 0.01) = 12.3 rad/s (measured 12.4-12.8),
# i.e. the 6.0 V datasheet speed. (A DC-motor actuator on top would count the
# back-EMF twice: 6.9 rad/s.)
XL330_NO_LOAD_SPEED = 123.0 * 2.0 * 3.141592653589793 / 60.0  # 12.88 rad/s

##
# Initial state.
##

# Default standing pose, taken from the default_joint_pos metadata of the
# walk.onnx policy shipped in the microban repo (MuJoCo joint frame).
DEFAULT_JOINT_POS: dict[str, float] = {
  "head": 0.0,
  "right_shoulder_pitch": 0.0,
  "right_shoulder_roll": -0.175,
  "right_elbow": -0.349,
  "right_hip_yaw": 0.0,
  "right_hip_roll": -0.087,
  "right_hip_pitch": -0.175,
  "right_knee": 0.0,
  "right_ankle_pitch": 0.0,
  "right_ankle_roll": 0.087,
  "left_shoulder_pitch": 0.0,
  "left_shoulder_roll": 0.175,
  "left_elbow": -0.349,
  "left_hip_yaw": 0.0,
  "left_hip_roll": 0.087,
  "left_hip_pitch": -0.175,
  "left_knee": 0.0,
  "left_ankle_pitch": 0.0,
  "left_ankle_roll": -0.087,
}

# Trunk height at which the feet just touch the ground in the default pose
# (the microban simulator spawns at 0.165, which starts ~7 mm inside the floor).
INIT_HEIGHT = 0.172

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, INIT_HEIGHT),
  joint_pos=DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

MICROBAN_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XL330_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_microban_robot_cfg() -> EntityCfg:
  """Return a fresh Microban EntityCfg (new instance to avoid shared mutation)."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=MICROBAN_ARTICULATION,
  )
