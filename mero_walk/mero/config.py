"""Settings with defaults; override any key in config.json next to this package.

Example config.json (only the keys you want to change):
    {"serial": {"baud": 1000000, "port": "/dev/ttyUSB0"},
     "battery": {"warn_v": 7.1}}
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "serial": {
        "port": "/dev/ttyAS0",      # Orange Pi Zero 2W UART on the Pollen Robot HAT
        "baud": 115200,             # all XL330 were set to 115200 (OPi cannot do 1 Mbps on this UART)
        "read_mode": "auto",        # auto | fast_indirect | fast | sync
    },
    "imu": {
        "i2c_bus": 1,
        "acc_addr": 0x19,           # BMI088 on the Pollen HAT (checked: chip id 0x1E)
        "gyro_addr": 0x68,          # (checked: chip id 0x0F)
        "rate_hz": 200.0,
    },
    "control": {
        "hz": 30.0,                 # walk.onnx was trained at 50 Hz, but 115200 baud tops out near 33 Hz
                                    # (see `python -m mero bench`); 50 needs a 1 Mbps USB adapter
        "kp_stand": 400,            # Position P Gain register while standing (XL330 default)
        "kp_walk": 125,             # while the policy runs (upstream KP_RL, ~0.28 Nm/rad)
        "ramp_s": 2.0,              # torque-on ramp to the stand pose
        "stop_settle_s": 1.0,       # keep the policy running at zero speed before standing still
        "fall_gravity_z": -0.5,     # projected gravity z above this = fallen (tilt > 60 deg)
        "torque_off_on_fall": True,
        "overcurrent_a": 15.0,      # summed estimated current that cuts torque (upstream value)
    },
    "policy": {
        "path": "models/walk.onnx",
        "gyro_frame": "imu",        # imu: upstream walk.onnx | body: policies whose base_ang_vel is in the trunk frame
    },
    "teleop": {
        "vx_step": 0.05, "vx_max": 0.30, "vx_min": -0.20,   # m/s
        "vy_step": 0.05, "vy_max": 0.15,                     # m/s
        "wz_step": 0.20, "wz_max": 1.00,                     # rad/s
    },
    "battery": {
        "cells": 2,
        "full_v": 8.4,
        "warn_v": 7.0,              # 3.5 V/cell: start thinking about charging
        "low_v": 6.8,               # 3.4 V/cell: walking is stopped and refused
        "critical_v": 6.4,          # 3.2 V/cell: torque off after a countdown
        "hysteresis_v": 0.1,
        "filter_tau_s": 8.0,        # voltage sags under load; levels use a filtered value
        "offset_v": 0.0,            # add to motor readings to match a multimeter at the battery
        "read_period_s": 1.0,       # one motor is read per period (rotating)
        "critical_countdown_s": 10.0,
        "state_file": "/tmp/mero-battery.json",
    },
    "stream": {"port": 9870, "hz": 20.0},
    "calibration_file": "calibration.json",
    "lock_file": "/tmp/mero-dxl.lock",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.json"
    cfg = copy.deepcopy(DEFAULTS)
    if p.exists():
        cfg = _merge(cfg, json.loads(p.read_text(encoding="utf-8")))
    return cfg


def resolve(rel: str) -> Path:
    p = Path(rel).expanduser()
    return p if p.is_absolute() else ROOT / p
