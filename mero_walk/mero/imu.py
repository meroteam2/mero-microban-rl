"""BMI088 on the Pollen Robot HAT (I2C) + attitude filter.

The chip frame -> trunk frame rotation comes from Calibration.mount_quat.
Outputs used by the walking policy:
  gyro_imu   : rad/s in the chip frame (what upstream walk.onnx expects)
  gyro_body  : rad/s in the trunk frame
  proj_grav  : unit gravity vector in the trunk frame ((0,0,-1) when upright)
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .calibration import quat_to_mat

# BMI088 registers
ACC_CHIP_ID, ACC_DATA, ACC_CONF, ACC_RANGE, ACC_PWR_CONF, ACC_PWR_CTRL, ACC_SOFTRESET = 0x00, 0x12, 0x40, 0x41, 0x7C, 0x7D, 0x7E
GYR_CHIP_ID, GYR_DATA, GYR_RANGE, GYR_BANDWIDTH, GYR_LPM1, GYR_SOFTRESET = 0x00, 0x02, 0x0F, 0x10, 0x11, 0x14


class BMI088:
    def __init__(self, bus: int = 1, acc_addr: int = 0x19, gyro_addr: int = 0x68):
        from smbus2 import SMBus
        self.i2c = SMBus(bus)
        self.acc_addr, self.gyro_addr = acc_addr, gyro_addr
        self._init()

    def _init(self) -> None:
        a, g, w = self.acc_addr, self.gyro_addr, self.i2c.write_byte_data
        w(a, ACC_SOFTRESET, 0xB6)
        time.sleep(0.002)
        self.i2c.read_byte_data(a, ACC_CHIP_ID)        # first access after reset
        cid = self.i2c.read_byte_data(a, ACC_CHIP_ID)
        if cid != 0x1E:
            raise RuntimeError(f"BMI088 accel chip id 0x{cid:02x} != 0x1e at 0x{a:02x}")
        w(a, ACC_PWR_CONF, 0x00)     # active
        time.sleep(0.005)
        w(a, ACC_PWR_CTRL, 0x04)     # accel on
        time.sleep(0.05)
        w(a, ACC_CONF, 0xA9)         # normal filter, 200 Hz ODR
        w(a, ACC_RANGE, 0x01)        # +-6 g
        self.acc_scale = 6.0 / 32768.0
        w(g, GYR_SOFTRESET, 0xB6)
        time.sleep(0.03)
        gid = self.i2c.read_byte_data(g, GYR_CHIP_ID)
        if gid != 0x0F:
            raise RuntimeError(f"BMI088 gyro chip id 0x{gid:02x} != 0x0f at 0x{g:02x}")
        w(g, GYR_RANGE, 0x01)        # +-1000 dps
        w(g, GYR_BANDWIDTH, 0x02)    # 1000 Hz ODR, 116 Hz filter
        w(g, GYR_LPM1, 0x00)         # normal mode
        self.gyro_scale = math.radians(1000.0) / 32768.0
        time.sleep(0.01)

    @staticmethod
    def _xyz(b: list[int]) -> np.ndarray:
        v = np.frombuffer(bytes(b), dtype="<i2").astype(float)
        return v

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """(acc in g, gyro in rad/s), both in the chip frame."""
        acc = self._xyz(self.i2c.read_i2c_block_data(self.acc_addr, ACC_DATA, 6)) * self.acc_scale
        gyr = self._xyz(self.i2c.read_i2c_block_data(self.gyro_addr, GYR_DATA, 6)) * self.gyro_scale
        return acc, gyr


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_two_vectors(a, b) -> np.ndarray:
    """Shortest rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    q = np.array([1.0 + d, *c])
    return q / np.linalg.norm(q)


class Mahony:
    """Mahony complementary filter; q maps chip-frame vectors to the world frame."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0):
        self.kp, self.ki = kp, ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.initialized = False

    def reset_from_acc(self, acc: np.ndarray) -> None:
        # world "up" (0,0,1) is measured as +acc direction in the chip frame
        self.q = quat_from_two_vectors(acc, [0.0, 0.0, 1.0])
        self.initialized = True

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        if not self.initialized:
            self.reset_from_acc(acc)
            return
        g = np.array(gyro, float)
        n = np.linalg.norm(acc)
        if 0.5 < n < 1.5:   # only trust the accelerometer near 1 g
            up_meas = acc / n
            R = quat_to_mat(self.q)
            up_est = R.T @ np.array([0.0, 0.0, 1.0])
            e = np.cross(up_meas, up_est)
            w = 1.0 - min(abs(n - 1.0) / 0.5, 1.0)
            kp = self.kp
            if np.linalg.norm(g) < 0.1 and abs(n - 1.0) < 0.05:
                kp *= 5.0   # robot held still: converge fast (calibration poses, after placing it)
            if self.ki > 0:
                self.integral += self.ki * e * dt
            g = g + w * kp * e + self.integral
        dq = 0.5 * quat_mul(self.q, [0.0, *g])
        self.q = self.q + dq * dt
        self.q /= np.linalg.norm(self.q)


@dataclass
class ImuSample:
    t: float
    acc_imu: np.ndarray        # g
    gyro_imu: np.ndarray       # rad/s, bias removed
    gyro_body: np.ndarray
    proj_grav: np.ndarray      # trunk frame
    quat_body: np.ndarray      # world <- trunk, (w,x,y,z), yaw from gyro integration
    valid: bool
    errors: int


class ImuFrames:
    """Chip -> trunk transforms (separate so the calibration UI can swap the mount live)."""

    def __init__(self, mount_quat, gyro_bias=(0, 0, 0)):
        self.set_mount(mount_quat)
        self.gyro_bias = np.array(gyro_bias, float)

    def set_mount(self, mount_quat) -> None:
        self.mount = np.array(mount_quat, float)
        self.R_mount = quat_to_mat(self.mount)
        self.mount_conj = self.mount * np.array([1, -1, -1, -1])

    def body(self, q_imu: np.ndarray) -> np.ndarray:
        return quat_mul(q_imu, self.mount_conj)


class ImuReader:
    """Reads a BMI088-like device on a background thread at rate_hz."""

    def __init__(self, device, frames: ImuFrames, rate_hz: float = 200.0):
        self.dev = device
        self.frames = frames
        self.period = 1.0 / rate_hz
        self.filter = Mahony()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="imu")
        self.errors = 0
        self._sample = ImuSample(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.array([0, 0, -1.0]),
                                 np.array([1.0, 0, 0, 0]), False, 0)

    def start(self) -> "ImuReader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def step(self, dt: float) -> None:
        """One read + filter update (called by the thread, or directly in tests)."""
        acc, gyr = self.dev.read()
        gyr = gyr - self.frames.gyro_bias
        self.filter.update(acc, gyr, dt)
        f = self.frames
        qb = f.body(self.filter.q)
        R = quat_to_mat(qb)
        pg = R.T @ np.array([0.0, 0.0, -1.0])
        s = ImuSample(time.perf_counter(), acc, gyr, f.R_mount @ gyr, pg, qb, True, self.errors)
        with self._lock:
            self._sample = s

    def _run(self) -> None:
        last = time.perf_counter()
        nxt = last
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = min(max(now - last, 1e-4), 0.05)
            last = now
            try:
                self.step(dt)
            except Exception:
                self.errors += 1
                time.sleep(0.01)
            nxt += self.period
            sl = nxt - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.perf_counter()

    def latest(self) -> ImuSample:
        with self._lock:
            return self._sample

    def average(self, seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Mean raw accelerometer (g) and gyro (rad/s, bias-corrected) over a window."""
        accs, gyrs = [], []
        t_end = time.perf_counter() + seconds
        last_t = -1.0
        while time.perf_counter() < t_end:
            s = self.latest()
            if s.valid and s.t != last_t:
                accs.append(s.acc_imu)
                gyrs.append(s.gyro_imu)
                last_t = s.t
            time.sleep(self.period)
        if not accs:
            raise RuntimeError("IMU 데이터가 들어오지 않습니다")
        return np.mean(accs, axis=0), np.mean(gyrs, axis=0)


def open_imu(cfg: dict, frames: ImuFrames) -> ImuReader:
    from .sim import backend
    if backend.enabled():
        return backend.open_imu(cfg, frames)
    c = cfg["imu"]
    dev = BMI088(int(c["i2c_bus"]), int(c["acc_addr"]), int(c["gyro_addr"]))
    return ImuReader(dev, frames, float(c["rate_hz"])).start()
