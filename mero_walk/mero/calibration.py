"""Joint sign/zero and IMU mounting calibration, stored in calibration.json.

Joint model (per motor):
    motor_rad = (raw - 2048) * 2*pi/4096          (same as rustypot / upstream)
    sim_rad   = sign * (motor_rad - zero)
    raw       = 2048 + (zero + sign * sim_rad) * 4096/(2*pi)

IMU model:
    mount_quat (w,x,y,z) = orientation of the IMU chip frame in the trunk frame,
    so a vector measured by the chip maps to the trunk as v_body = R(mount) v_imu.
    Upstream value (0.5,-0.5,-0.5,0.5) = MJCF site "imu".
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .joints import DEFAULT_SIGN, JOINTS

TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)
RAD_PER_TICK = 1.0 / TICKS_PER_RAD
VEL_UNIT = 0.229 * 2.0 * math.pi / 60.0   # rad/s per Present Velocity unit
UPSTREAM_MOUNT = (0.5, -0.5, -0.5, 0.5)


@dataclass
class JointCal:
    sign: float = 1.0
    zero: float = 0.0   # rad, in motor frame

    def to_sim(self, raw: float) -> float:
        return self.sign * ((raw - 2048.0) * RAD_PER_TICK - self.zero)

    def to_raw(self, q: float) -> int:
        return int(round(2048.0 + (self.zero + self.sign * q) * TICKS_PER_RAD))

    def vel_to_sim(self, raw_vel: float) -> float:
        return self.sign * raw_vel * VEL_UNIT


@dataclass
class Calibration:
    joints: dict[str, JointCal] = field(default_factory=lambda: {n: JointCal(DEFAULT_SIGN[n], 0.0) for n in JOINTS})
    mount_quat: tuple[float, float, float, float] = UPSTREAM_MOUNT
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)   # rad/s, IMU frame
    path: Path | None = None

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        p = Path(path)
        cal = cls(path=p)
        if not p.exists():
            return cal
        d = json.loads(p.read_text(encoding="utf-8"))
        for name, jd in d.get("joints", {}).items():
            if name in cal.joints:
                cal.joints[name] = JointCal(float(jd.get("sign", DEFAULT_SIGN[name])),
                                            math.radians(float(jd.get("zero_deg", 0.0))))
        imu = d.get("imu", {})
        if "mount_quat" in imu:
            cal.mount_quat = tuple(float(v) for v in imu["mount_quat"])
        if "gyro_bias" in imu:
            cal.gyro_bias = tuple(float(v) for v in imu["gyro_bias"])
        return cal

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joints": {n: {"sign": int(c.sign), "zero_deg": round(math.degrees(c.zero), 3),
                           "default_sign": int(DEFAULT_SIGN[n])} for n, c in self.joints.items()},
            "imu": {"mount_quat": [round(v, 6) for v in self.mount_quat],
                    "gyro_bias": [round(v, 6) for v in self.gyro_bias]},
        }

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no calibration path")
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.path = p
        return p

    def diff_from_default(self) -> list[str]:
        out = []
        for n, c in self.joints.items():
            if c.sign != DEFAULT_SIGN[n]:
                out.append(f"{n}: 부호 반전됨 (기본 {int(DEFAULT_SIGN[n]):+d} → {int(c.sign):+d})")
            if abs(c.zero) > 1e-6:
                out.append(f"{n}: 원점 {math.degrees(c.zero):+.1f}°")
        if any(abs(a - b) > 1e-6 for a, b in zip(self.mount_quat, UPSTREAM_MOUNT)):
            out.append(f"IMU 장착 회전 {tuple(round(v, 3) for v in self.mount_quat)} (기본 {UPSTREAM_MOUNT})")
        return out


# ---------------------------------------------------------------------- quaternion helpers
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def _axis_rotations() -> list[np.ndarray]:
    mats = []
    eye = np.eye(3)
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for r, c in enumerate(perm):
                R[r, c] = signs[r]
            if np.linalg.det(R) > 0:
                mats.append(R)
    assert len(mats) == 24 and any(np.allclose(m, eye) for m in mats)
    return mats


AXIS_ROTATIONS = _axis_rotations()

# Known poses for the IMU wizard: gravity direction expressed in the trunk frame
# (x forward, y left, z up).
IMU_POSES = {
    "upright": ((0.0, 0.0, -1.0), "똑바로 세우기 (서 있는 자세 그대로)"),
    "face_down": ((1.0, 0.0, 0.0), "앞으로 엎드려 눕히기 (배가 바닥, 얼굴이 아래)"),
    "left_down": ((0.0, 1.0, 0.0), "왼쪽 옆구리가 바닥에 닿게 눕히기"),
}


def solve_mount(samples: dict[str, np.ndarray]) -> dict:
    """Find R with g_body = R g_imu from accelerometer samples in known poses.

    samples: pose name -> mean accelerometer reading (g) in the IMU frame.
    Gravity direction in the IMU frame is -acc (the chip measures specific force).
    Returns dict(quat, err_deg, snapped, improper, per_pose).
    """
    B, A = [], []
    for name, acc in samples.items():
        g_imu = -np.asarray(acc, dtype=float)
        g_imu /= max(np.linalg.norm(g_imu), 1e-9)
        A.append(g_imu)
        B.append(np.array(IMU_POSES[name][0]))
    A, B = np.array(A), np.array(B)
    # Wahba / Kabsch: maximise sum b_i . R a_i
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    # an improper fit would be clearly better if an axis is mirrored in the driver
    R_improper = Vt.T @ np.diag([1, 1, -d]) @ U.T

    def err(Rm):
        return [math.degrees(math.acos(np.clip(np.dot(Rm @ a, b), -1, 1))) for a, b in zip(A, B)]

    best_axis = min(AXIS_ROTATIONS, key=lambda M: max(err(M)))
    snapped = max(err(best_axis)) < 15.0
    R_use = best_axis if snapped else R
    e = err(R_use)
    improper = (len(samples) >= 3 and max(err(R_improper)) + 5.0 < max(err(R)))
    return {
        "quat": mat_to_quat(R_use),
        "err_deg": max(e),
        "snapped": snapped,
        "improper": improper,
        "per_pose": dict(zip(samples.keys(), e)),
        "R": R_use,
    }
