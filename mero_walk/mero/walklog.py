"""Per-walk CSV log: one row per control tick from walk start until the robot stands/stops.

    logs/walk_YYYYmmdd_HHMMSS.csv    pandas.read_csv(path)
    logs/walk_YYYYmmdd_HHMMSS.json   policy/gain/calibration context of that walk

Columns: time and state, commands, IMU, all 19 joint angles/velocities (sim convention),
the policy's observation vector and raw action, the joint targets written to the motors,
battery voltage, read errors and any status message ("event") emitted on that tick.
Rows are flushed every second and fsynced on close (power can drop without warning).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .joints import JOINTS


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return '"' + v.replace('"', "'") + '"'
    f = float(v)
    return "" if math.isnan(f) else f"{f:.6g}"


class WalkLog:
    def __init__(self, directory: str | Path, enabled: bool = True):
        self.dir = Path(directory)
        self.enabled = enabled
        self.f = None
        self.path: Path | None = None
        self.rows = 0
        self._t0 = 0.0
        self._last_flush = 0.0
        self._events: list[str] = []
        self._order: list[str] = []
        self._obs_n = 0

    @property
    def active(self) -> bool:
        return self.f is not None

    def start(self, order: list[str], obs_n: int, meta: dict) -> Path | None:
        if not self.enabled:
            return None
        self.close("restart")
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"walk_{stamp}.csv"
        k = 1
        while self.path.exists():
            self.path = self.dir / f"walk_{stamp}_{k}.csv"
            k += 1
        self._order, self._obs_n = list(order), int(obs_n)
        self.path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
        cols = (["t", "state", "cmd_des_vx", "cmd_des_vy", "cmd_des_wz", "cmd_vx", "cmd_vy", "cmd_wz",
                 "imu_age_ms", "imu_valid"]
                + [f"acc_{a}" for a in "xyz"] + [f"gyro_imu_{a}" for a in "xyz"]
                + [f"gyro_body_{a}" for a in "xyz"] + [f"grav_{a}" for a in "xyz"]
                + [f"quat_{a}" for a in "wxyz"]
                + [f"q_{n}" for n in JOINTS] + [f"dq_{n}" for n in JOINTS]
                + [f"obs_{i}" for i in range(self._obs_n)]
                + [f"act_{n}" for n in self._order] + [f"tgt_{n}" for n in self._order]
                + ["volt", "read_errors", "event"])
        self.f = open(self.path, "w", encoding="utf-8", buffering=1 << 16)
        self.f.write(",".join(cols) + "\n")
        self.rows = 0
        self._t0 = self._last_flush = time.monotonic()
        self._events.clear()
        return self.path

    def event(self, msg: str) -> None:
        if self.f is not None:
            self._events.append(str(msg))

    def row(self, state: str, cmd_des, cmd, imu, q: dict, dq: dict, obs, act, tgt: dict | None,
            volt: float | None, read_errors: int) -> None:
        if self.f is None:
            return
        now = time.monotonic()
        nan3, nan4 = [math.nan] * 3, [math.nan] * 4
        age = (time.perf_counter() - imu.t) * 1000.0 if imu is not None else math.nan
        vals = ([now - self._t0, state] + list(cmd_des) + list(cmd)
                + [age, float(bool(imu.valid)) if imu is not None else math.nan]
                + (list(imu.acc_imu) if imu is not None else nan3)
                + (list(imu.gyro_imu) if imu is not None else nan3)
                + (list(imu.gyro_body) if imu is not None else nan3)
                + (list(imu.proj_grav) if imu is not None else nan3)
                + (list(imu.quat_body) if imu is not None else nan4)
                + [q.get(n, math.nan) for n in JOINTS] + [dq.get(n, math.nan) for n in JOINTS]
                + (list(np.asarray(obs, float)) if obs is not None else [math.nan] * self._obs_n)
                + (list(np.asarray(act, float)) if act is not None else [math.nan] * len(self._order))
                + [(tgt or {}).get(n, math.nan) for n in self._order]
                + [volt if volt is not None else math.nan, read_errors, " | ".join(self._events)])
        self._events.clear()
        self.f.write(",".join(_fmt(v) for v in vals) + "\n")
        self.rows += 1
        if now - self._last_flush > 1.0:
            self._last_flush = now
            self.f.flush()

    def close(self, why: str = "") -> Path | None:
        if self.f is None:
            return None
        f, self.f = self.f, None
        try:
            f.flush()
            os.fsync(f.fileno())
        finally:
            f.close()
        return self.path
