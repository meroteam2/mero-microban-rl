"""MERO_SIM=1 python -m mero run|calib  ->  same apps, MuJoCo robot instead of hardware.

MERO_SIM=held keeps the trunk fixed in the air (handy for `calib` jogging).
"""
from __future__ import annotations

import os
import time

_PLANT = None


def enabled() -> bool:
    return bool(os.environ.get("MERO_SIM"))


def plant(cfg):
    global _PLANT
    if _PLANT is None:
        from .plant import Plant
        _PLANT = Plant(control_hz=float(cfg["control"]["hz"]), fixed_base=os.environ.get("MERO_SIM") == "held")
        p = _PLANT
        p._last_wall = time.perf_counter()
        if os.environ.get("MERO_SIM_BAUD"):          # emulate wire time of a real bus
            p.ser.baud = int(os.environ["MERO_SIM_BAUD"])

        def before_read(p=p):
            # keep time flowing for joints with torque on even if nobody writes goals (jog)
            now = time.perf_counter()
            if now - p._last_wall > p.dt_ctrl and any(m.torque for m in p.motors.values()):
                p.step(min(now - p._last_wall, 0.1))
            p._last_wall = now
            p._sync_present()
        p.ser.before_read = before_read
        orig = p._on_write

        def on_write(addr, written, p=p, orig=orig):
            orig(addr, written)
            p._last_wall = time.perf_counter()
        p.ser.on_write = on_write
    return _PLANT


def open_bus(cfg, cal):
    from ..bus import MotorBus
    from ..dxl import Protocol2
    return MotorBus(Protocol2(ser=plant(cfg).ser, baud=1_000_000), cal,
                    multiturn=cfg["serial"].get("multiturn_joints", []))


def open_imu(cfg, frames):
    from ..imu import ImuReader
    from .plant import FakeImuDevice
    return ImuReader(FakeImuDevice(plant(cfg)), frames, float(cfg["imu"]["rate_hz"])).start()
