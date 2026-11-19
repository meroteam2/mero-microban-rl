"""Joint names, Dynamixel IDs and conventions for the MERO Microban.

Values are taken from Rhoban/microban (src/constants.py, src/model/mjcf/robot.xml,
commit d0a128d). Joint angles everywhere in this package are in the *simulation*
convention (the MJCF joints the walking policy was trained on). The mapping to the
motor's raw position lives in calibration.py.
"""
from __future__ import annotations

import math

# Order used for every 19-vector in this package (state streaming, calibration UI).
JOINTS: list[str] = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "head",
]

MOTOR_ID: dict[str, int] = {
    "left_hip_yaw": 11, "left_hip_roll": 12, "left_hip_pitch": 13, "left_knee": 14,
    "left_ankle_pitch": 15, "left_ankle_roll": 16,
    "right_hip_yaw": 21, "right_hip_roll": 22, "right_hip_pitch": 23, "right_knee": 24,
    "right_ankle_pitch": 25, "right_ankle_roll": 26,
    "left_shoulder_pitch": 31, "left_shoulder_roll": 32, "left_elbow": 33,
    "right_shoulder_pitch": 41, "right_shoulder_roll": 42, "right_elbow": 43,
    "head": 51,
}
ID_TO_JOINT: dict[int, str] = {v: k for k, v in MOTOR_ID.items()}

# Korean labels for the terminal UI (robot's own left/right).
LABEL_KO: dict[str, str] = {
    "left_hip_yaw": "왼 엉덩이 Yaw", "left_hip_roll": "왼 엉덩이 Roll", "left_hip_pitch": "왼 엉덩이 Pitch",
    "left_knee": "왼 무릎", "left_ankle_pitch": "왼 발목 Pitch", "left_ankle_roll": "왼 발목 Roll",
    "right_hip_yaw": "오른 엉덩이 Yaw", "right_hip_roll": "오른 엉덩이 Roll", "right_hip_pitch": "오른 엉덩이 Pitch",
    "right_knee": "오른 무릎", "right_ankle_pitch": "오른 발목 Pitch", "right_ankle_roll": "오른 발목 Roll",
    "left_shoulder_pitch": "왼 어깨 Pitch", "left_shoulder_roll": "왼 어깨 Roll", "left_elbow": "왼 팔꿈치",
    "right_shoulder_pitch": "오른 어깨 Pitch", "right_shoulder_roll": "오른 어깨 Roll", "right_elbow": "오른 팔꿈치",
    "head": "머리",
}

# Default hardware direction: sim_angle = SIGN * motor_angle (upstream MOTOR_SIGN).
DEFAULT_SIGN: dict[str, float] = {
    "left_hip_yaw": -1.0, "left_hip_roll": 1.0, "left_hip_pitch": -1.0, "left_knee": 1.0,
    "left_ankle_pitch": 1.0, "left_ankle_roll": -1.0,
    "right_hip_yaw": -1.0, "right_hip_roll": 1.0, "right_hip_pitch": 1.0, "right_knee": -1.0,
    "right_ankle_pitch": -1.0, "right_ankle_roll": -1.0,
    "left_shoulder_pitch": 1.0, "left_shoulder_roll": -1.0, "left_elbow": 1.0,
    "right_shoulder_pitch": -1.0, "right_shoulder_roll": -1.0, "right_elbow": -1.0,
    "head": 1.0,
}

_d = math.radians
# Upstream NEUTRAL_POSE: the standing pose the robot ramps to after torque-on.
STAND_POSE: dict[str, float] = {
    "left_hip_yaw": 0.0, "left_hip_roll": _d(5.0), "left_hip_pitch": _d(-10.0), "left_knee": 0.0,
    "left_ankle_pitch": 0.0, "left_ankle_roll": _d(-5.0),
    "right_hip_yaw": 0.0, "right_hip_roll": _d(-5.0), "right_hip_pitch": _d(-10.0), "right_knee": 0.0,
    "right_ankle_pitch": 0.0, "right_ankle_roll": _d(5.0),
    "left_shoulder_pitch": _d(10.0), "left_shoulder_roll": _d(10.0), "left_elbow": _d(-20.0),
    "right_shoulder_pitch": _d(10.0), "right_shoulder_roll": _d(-10.0), "right_elbow": _d(-20.0),
    "head": 0.0,
}

# All-zero pose of the MJCF: legs straight, arms hanging straight against the body,
# head facing forward. This is the reference pose used for zero-offset calibration.
ZERO_POSE: dict[str, float] = {n: 0.0 for n in JOINTS}

# MJCF joint ranges (rad, sim convention).
JOINT_RANGE: dict[str, tuple[float, float]] = {
    "left_hip_yaw": (-1.0472, 4.1888),
    "left_hip_roll": (-0.6109, 0.6109),
    "left_hip_pitch": (-1.5708, 1.5708),
    "left_knee": (-0.7854, 2.3562),
    "left_ankle_pitch": (-1.5708, 0.6109),
    "left_ankle_roll": (-0.6109, 0.6109),
    "right_hip_yaw": (-4.1888, 1.0472),
    "right_hip_roll": (-0.6109, 0.6109),
    "right_hip_pitch": (-1.5708, 1.5708),
    "right_knee": (-0.7854, 2.3562),
    "right_ankle_pitch": (-1.5708, 0.6109),
    "right_ankle_roll": (-0.6109, 0.6109),
    "left_shoulder_pitch": (-3.1416, 3.1416),
    "left_shoulder_roll": (0.0, 3.1416),
    "left_elbow": (-2.1817, 2.1817),
    "right_shoulder_pitch": (-3.1416, 3.1416),
    "right_shoulder_roll": (-3.1416, 0.0),
    "right_elbow": (-2.1817, 2.1817),
    "head": (-1.5708, 1.5708),
}

# Order of the 18 actuated joints in walk.onnx observations/actions (head excluded).
POLICY_ORDER_DEFAULT: list[str] = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
]

# Mirror pairs, used by the calibration UI to compare left/right.
MIRROR: dict[str, str] = {}
for _n in JOINTS:
    if _n.startswith("left_"):
        MIRROR[_n] = "right_" + _n[5:]
        MIRROR["right_" + _n[5:]] = _n
