"""High-level access to the 19 XL330 motors in simulation-convention radians."""
from __future__ import annotations

import fcntl
import os
import struct
import time
from pathlib import Path

import math

import numpy as np

from . import dxl
from .calibration import Calibration, JointCal
from .joints import JOINTS, JOINT_RANGE, MOTOR_ID

PRESENT_BLOCK = (dxl.ADDR_PRESENT_VELOCITY, 8)   # velocity(4) + position(4), contiguous


def turn_ticks(c: JointCal, name: str, raw: float) -> int:
    """Whole turns (in ticks) to subtract from a multi-turn motor's raw position so the joint
    angle lands in the turn nearest the middle of its range."""
    lo, hi = JOINT_RANGE[name]
    k = round((c.to_sim(raw) - (lo + hi) / 2) / (2 * math.pi))
    return int(k * c.sign) * 4096


def _upstream_running() -> bool:
    """Rhoban's own control loop (src/main.py) keeps a PID file instead of our lock."""
    try:
        pid = int(Path("/tmp/microban_scheduler.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


class BusLock:
    """Only one program may own the motor bus (two writers make motors jump)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.fd = None

    def acquire(self, wait: bool = False) -> bool:
        writable = True
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                os.fchmod(self.fd, 0o666)   # root service and the orangepi user share it
            except OSError:
                pass
        except PermissionError:
            self.fd = os.open(self.path, os.O_RDONLY)
            writable = False
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            return False
        if writable:
            os.ftruncate(self.fd, 0)
            os.write(self.fd, f"{os.getpid()}\n".encode())
        if _upstream_running():
            self.release()
            return False
        return True

    def owner(self) -> str:
        try:
            return self.path.read_text().strip()
        except OSError:
            return "?"

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class MotorBus:
    def __init__(self, proto: dxl.Protocol2, cal: Calibration, joints: list[str] | None = None,
                 multiturn: list[str] | tuple = ()):
        self.p = proto
        self.cal = cal
        self.joints = list(joints or JOINTS)
        # Joints whose reference pose sits at the motor's 0/4095 wrap (horn mounted ~180° off).
        # Their motors run in Extended Position mode (4): in Position mode (3) the motor re-wraps
        # its position into 0..4095 when torque turns on, so a goal computed just before is a
        # full turn away and the joint spins 360°. Here the bus tracks the turn instead.
        self.multiturn = set(multiturn)
        self.turn: dict[str, int] = {n: 0 for n in self.multiturn}
        self._torque_on: set[str] = set()
        self.ids = [MOTOR_ID[n] for n in self.joints]
        self.read_mode = "sync"
        self._ind_addr = None
        self.firmware: dict[int, int] = {}
        self.models: dict[int, int] = {}
        self.last_ok = 0.0

    # ------------------------------------------------------------------ setup
    def scan(self) -> dict[int, tuple[int, int]]:
        found = {}
        for mid in self.ids:
            r = self.p.ping(mid)
            if r is None:
                r = self.p.ping(mid)  # one retry
            if r is not None:
                found[mid] = r
                self.models[mid], self.firmware[mid] = r
        return found

    def setup(self, read_mode: str = "auto") -> str:
        """Ping everything and choose the fastest working read method."""
        found = self.scan()
        missing = [f"{n}({MOTOR_ID[n]})" for n in self.joints if MOTOR_ID[n] not in found]
        if missing:
            raise dxl.DxlError("응답 없는 모터: " + ", ".join(missing))
        for n in sorted(self.multiturn):
            mode = self.p.read(MOTOR_ID[n], dxl.ADDR_OPERATING_MODE, 1)[0]
            if mode != 4:
                raise dxl.DxlError(f"{n}({MOTOR_ID[n]}) Operating Mode {mode}: 다회전 관절은 4(확장 위치)여야 합니다. "
                                   "`mero check --fix`로 바꾸세요")
        fws = {self.firmware[i] for i in self.ids}
        fast_ok = min(fws) >= 46
        if read_mode in ("auto", "fast_indirect") and fast_ok and self._setup_indirect():
            self.read_mode = "fast_indirect"
        elif read_mode in ("auto", "fast", "fast_indirect") and fast_ok:
            self.read_mode = "fast"
        else:
            self.read_mode = "sync"
        # verify once
        self.read_state()
        return self.read_mode

    def _setup_indirect(self) -> bool:
        """Map position(lo,hi)+velocity(lo,hi) to 4 contiguous indirect bytes.

        Positions stay near 0..4095 (multi-turn joints within a few turns) and velocities well
        inside int16, so 16 bits each are enough and halve the bytes on the slow bus.
        """
        fws = {self.firmware[i] for i in self.ids}
        if all(f >= 53 for f in fws):
            data_addr, old_slot = dxl.ADDR_INDIRECT_DATA_1_NEW, 0
        elif all(f < 53 for f in fws):
            data_addr, old_slot = dxl.ADDR_INDIRECT_DATA_1_OLD, 0
        else:
            # Mixed V52/V53 chain: V53 moved Indirect Data 1 from 208 to 224. Old-firmware motors
            # use slots 17..20 instead, whose data (208+16 = 224..227) lines up with the new
            # motors' slots 1..4, so one Fast Sync Read at 224 still covers everyone.
            data_addr = dxl.ADDR_INDIRECT_DATA_1_NEW
            old_slot = (dxl.ADDR_INDIRECT_DATA_1_NEW - dxl.ADDR_INDIRECT_DATA_1_OLD)
        src = [dxl.ADDR_PRESENT_POSITION, dxl.ADDR_PRESENT_POSITION + 1,
               dxl.ADDR_PRESENT_VELOCITY, dxl.ADDR_PRESENT_VELOCITY + 1]
        try:
            new_ids = [m for m in self.ids if self.firmware[m] >= 53 or not old_slot]
            old_ids = [m for m in self.ids if m not in new_ids]
            for k, a in enumerate(src):
                for ids, slot in ((new_ids, 0), (old_ids, old_slot)):
                    if ids:
                        self.p.sync_write(dxl.ADDR_INDIRECT_ADDRESS_1 + 2 * (slot + k), 2,
                                          [(mid, dxl.u16(a)) for mid in ids])
                        time.sleep(0.002)
            # read back: indirect data must equal the direct registers
            direct = self.p.sync_read(*PRESENT_BLOCK, self.ids)
            ind = self.p.fast_sync_read(data_addr, 4, self.ids)
            for mid in self.ids:
                pos_d = struct.unpack("<i", direct[mid][4:8])[0]
                pos_i = struct.unpack("<h", ind[mid][0:2])[0]
                if abs(pos_d - pos_i) > 20:   # motors may move slightly between the reads
                    return False
        except dxl.DxlError:
            return False
        self._ind_addr = data_addr
        return True

    # ------------------------------------------------------------------ state
    def read_raw(self, joints: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Raw positions (ticks) and velocities (0.229 rpm units) in `joints` order."""
        ids = self.ids if joints is None else [MOTOR_ID[n] for n in joints]
        if self.read_mode == "fast_indirect":
            d = self.p.fast_sync_read(self._ind_addr, 4, ids)
            pos = [struct.unpack("<h", d[i][0:2])[0] for i in ids]
            vel = [struct.unpack("<h", d[i][2:4])[0] for i in ids]
        else:
            fn = self.p.fast_sync_read if self.read_mode == "fast" else self.p.sync_read
            d = fn(*PRESENT_BLOCK, ids)
            vel = [struct.unpack("<i", d[i][0:4])[0] for i in ids]
            pos = [struct.unpack("<i", d[i][4:8])[0] for i in ids]
        self.last_ok = time.perf_counter()
        return np.array(pos, dtype=float), np.array(vel, dtype=float)

    def read_state(self, joints: list[str] | None = None) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
        """Joint angles/velocities in sim convention. `joints` = subset to save bus time."""
        joints = self.joints if joints is None else joints
        raw_pos, raw_vel = self.read_raw(joints)
        q, dq = {}, {}
        for k, n in enumerate(joints):
            c = self.cal.joints[n]
            raw = raw_pos[k]
            if n in self.multiturn:
                if n not in self._torque_on:      # frozen while torque is on (goals use it)
                    self.turn[n] = turn_ticks(c, n, raw)
                raw = raw - self.turn[n]
            q[n] = c.to_sim(raw)
            dq[n] = c.vel_to_sim(raw_vel[k])
        return q, dq, raw_pos

    # ------------------------------------------------------------------ commands
    def clamp(self, name: str, q: float) -> float:
        lo, hi = JOINT_RANGE[name]
        return min(max(q, lo), hi)

    def write_goal(self, targets: dict[str, float]) -> None:
        items = []
        for n, q in targets.items():
            raw = self.cal.joints[n].to_raw(self.clamp(n, q))
            if n in self.multiturn:
                raw += self.turn[n]               # Extended Position mode takes any turn
            else:
                raw = min(max(raw, 0), 4095)
            items.append((MOTOR_ID[n], dxl.s32(raw)))
        if items:
            self.p.sync_write(dxl.ADDR_GOAL_POSITION, 4, items)

    def torque(self, on: bool, joints: list[str] | None = None) -> None:
        names = list(joints or self.joints)
        ids = [MOTOR_ID[n] for n in names]
        if on:
            self._torque_on.update(names)
        else:
            self._torque_on.difference_update(names)
        self.p.sync_write(dxl.ADDR_TORQUE_ENABLE, 1, [(i, bytes([1 if on else 0])) for i in ids])

    def torque_off_all_robust(self) -> None:
        """Best effort torque-off used on every exit path (sent twice)."""
        self._torque_on.clear()
        for _ in range(2):
            try:
                self.p.sync_write(dxl.ADDR_TORQUE_ENABLE, 1, [(MOTOR_ID[n], b"\x00") for n in JOINTS])
            except Exception:
                pass
            time.sleep(0.01)

    def set_kp(self, kp: int, joints: list[str] | None = None) -> None:
        ids = [MOTOR_ID[n] for n in (joints or self.joints)]
        self.p.sync_write(dxl.ADDR_POSITION_P, 2, [(i, dxl.u16(kp)) for i in ids])

    def no_profile(self) -> None:
        """Profile velocity/acceleration 0 = goal positions are tracked immediately."""
        self.p.sync_write(dxl.ADDR_PROFILE_ACCEL, 4, [(i, dxl.u32(0)) for i in self.ids])
        self.p.sync_write(dxl.ADDR_PROFILE_VELOCITY, 4, [(i, dxl.u32(0)) for i in self.ids])

    def led(self, on: bool, joints: list[str] | None = None) -> None:
        ids = [MOTOR_ID[n] for n in (joints or JOINTS)]
        self.p.sync_write(dxl.ADDR_LED, 1, [(i, bytes([1 if on else 0])) for i in ids])

    def read_voltage(self, joint: str) -> float:
        d = self.p.read(MOTOR_ID[joint], dxl.ADDR_PRESENT_VOLTAGE, 2)
        return (d[0] | (d[1] << 8)) * 0.1

    def read_torque_enabled(self) -> dict[str, bool]:
        d = self.p.sync_read(dxl.ADDR_TORQUE_ENABLE, 1, self.ids)
        return {n: bool(d[MOTOR_ID[n]][0]) for n in self.joints}


def open_bus(cfg: dict, cal: Calibration, ser=None) -> MotorBus:
    from .sim import backend
    if backend.enabled() and ser is None:
        return backend.open_bus(cfg, cal)
    proto = dxl.Protocol2(cfg["serial"]["port"], int(cfg["serial"]["baud"]), ser=ser)
    return MotorBus(proto, cal, multiturn=cfg["serial"].get("multiturn_joints", []))
