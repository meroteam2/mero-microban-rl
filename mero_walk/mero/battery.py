"""2S Li-ion battery protection from the XL330 'Present Input Voltage' register.

Voltage sags while walking, so levels use a low-pass filtered value; a reading far
below `critical_v` for several samples in a row also counts (sudden drop).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

OK, WARN, LOW, CRITICAL = 0, 1, 2, 3
LEVEL_NAME = {OK: "OK", WARN: "WARN", LOW: "LOW", CRITICAL: "CRITICAL"}
LEVEL_KO = {OK: "정상", WARN: "주의", LOW: "부족", CRITICAL: "위험"}


@dataclass
class BatteryStatus:
    level: int = OK
    filtered_v: float = float("nan")
    last_v: float = float("nan")
    min_v: float = float("nan")
    percent: float = float("nan")
    samples: int = 0
    changed: bool = False


def soc_percent(v_pack: float, cells: int) -> float:
    """Rough rest-voltage state of charge for Li-ion (only a hint)."""
    table = [(3.0, 0), (3.3, 5), (3.5, 15), (3.6, 30), (3.7, 45), (3.8, 60), (3.9, 72), (4.0, 83), (4.1, 92), (4.2, 100)]
    v = v_pack / cells
    if v <= table[0][0]:
        return 0.0
    for (v0, p0), (v1, p1) in zip(table, table[1:]):
        if v <= v1:
            return p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 100.0


class BatteryMonitor:
    def __init__(self, cfg: dict):
        c = cfg["battery"]
        self.cells = int(c["cells"])
        self.th = {WARN: float(c["warn_v"]), LOW: float(c["low_v"]), CRITICAL: float(c["critical_v"])}
        self.hyst = float(c["hysteresis_v"])
        self.tau = float(c["filter_tau_s"])
        self.offset = float(c["offset_v"])
        self.state_file = c.get("state_file")
        self.status = BatteryStatus()
        self._t_last = None
        self._below_crit = 0
        self._recent: list[float] = []

    def _level_for(self, v: float, current: int) -> int:
        lvl = OK
        for L in (WARN, LOW, CRITICAL):
            if v < self.th[L]:
                lvl = L
        # hysteresis: only go back up once clearly above the threshold we are in
        if lvl < current and v < self.th[current] + self.hyst:
            return current
        return lvl

    def update(self, volts: float, t: float | None = None) -> BatteryStatus:
        t = time.monotonic() if t is None else t
        v = volts + self.offset
        s = self.status
        if math.isnan(s.filtered_v) or self._t_last is None:
            s.filtered_v = v
        else:
            a = 1.0 - math.exp(-max(t - self._t_last, 0.0) / self.tau)
            s.filtered_v += a * (v - s.filtered_v)
        self._t_last = t
        s.last_v = v
        self._recent = (self._recent + [v])[-20:]
        s.min_v = min(self._recent)
        s.samples += 1
        s.percent = soc_percent(s.filtered_v, self.cells)
        self._below_crit = self._below_crit + 1 if v < self.th[CRITICAL] - 0.2 else 0
        new = self._level_for(s.filtered_v, s.level)
        if self._below_crit >= 3:
            new = CRITICAL
        s.changed = new != s.level
        s.level = new
        self._write_state(t)
        return s

    def _write_state(self, t: float) -> None:
        if not self.state_file:
            return
        s = self.status
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"time": time.time(), "pid": os.getpid(), "level": LEVEL_NAME[s.level],
                           "filtered_v": round(s.filtered_v, 2), "last_v": round(s.last_v, 2),
                           "percent": round(s.percent)}, f)
            os.replace(tmp, self.state_file)
        except OSError:
            pass

    def line(self) -> str:
        s = self.status
        if s.samples == 0:
            return "배터리 ?.?V"
        return f"배터리 {s.filtered_v:.2f}V(~{s.percent:.0f}%) {LEVEL_KO[s.level]}"
