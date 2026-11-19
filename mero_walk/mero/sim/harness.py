"""Run the real runtime stack against the MuJoCo plant (no hardware).

    python -m mero.sim.harness --seconds 12 --vx 0.15 --hz 50
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .. import config, term
from ..battery import BatteryMonitor
from ..bus import MotorBus
from ..calibration import Calibration
from ..controller import STAND, WALK, IDLE, WalkController
from ..dxl import Protocol2
from ..imu import ImuFrames, ImuReader
from ..policy import WalkPolicy
from .plant import FakeImuDevice, Plant


class SimRobot:
    def __init__(self, hz: float = 50.0, true_cal: Calibration | None = None, sw_cal: Calibration | None = None,
                 echo: bool = False, vin: float = 7.6, gyro_frame: str = "imu", policy_path: str | None = None,
                 true_mount=None, quiet: bool = True, act_delay_s: float = 0.0):
        self.cfg = config.load("/nonexistent")
        self.cfg["control"]["hz"] = hz
        self.cfg["battery"]["state_file"] = None
        self.cfg["log"]["walk"] = False        # tests opt in with their own WalkLog
        self.plant = Plant(true_cal=true_cal, control_hz=hz, vin=vin, act_delay_s=act_delay_s)
        self.plant.ser.echo = echo
        self.cal = sw_cal or Calibration()
        self.bus = MotorBus(Protocol2(ser=self.plant.ser), self.cal)
        self.mode = self.bus.setup("auto")
        dev = FakeImuDevice(self.plant, true_mount=true_mount)
        self.imu = ImuReader(dev, ImuFrames(self.cal.mount_quat, self.cal.gyro_bias))
        pol = WalkPolicy(policy_path or str(config.resolve(self.cfg["policy"]["path"])), gyro_frame)
        self.mon = BatteryMonitor(self.cfg)
        self.events: list[str] = []
        emit = (lambda m: self.events.append(m)) if quiet else (lambda m: (self.events.append(m), print(m)))
        self.ctl = WalkController(self.cfg, self.bus, self.imu, pol, self.mon, emit=emit,
                                  clock=lambda: self.plant.sim_time)
        self.dt = 1.0 / hz
        self.imu_frozen = False

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            # IMU thread replacement: 4 filter updates per control tick
            for _ in range(0 if self.imu_frozen else 4):
                self.imu.step(self.dt / 4)
            before = self.plant.sim_time
            self.ctl.tick()
            if self.plant.sim_time == before:
                # nothing written (torque off): the robot is "held" by a person, so only the
                # clock advances (otherwise it would collapse before the test starts)
                self.plant.sim_time += self.dt

    def key(self, k: str) -> None:
        self.ctl.handle_key(k)

    def run_until(self, cond, max_s: float) -> bool:
        t_end = self.plant.sim_time + max_s
        while self.plant.sim_time < t_end:
            self.tick()
            if cond():
                return True
        return False

    def stand_up(self) -> None:
        self.tick(3)
        self.key("t")
        assert self.run_until(lambda: self.ctl.state == STAND, 5.0), self.events

    def walk(self, vx: float, seconds: float, vy: float = 0.0, wz: float = 0.0) -> dict:
        tp = self.cfg["teleop"]
        self.key(term.KEY_SPACE)
        assert self.ctl.state == WALK, self.events
        self.ctl.cmd_des[:] = [vx, vy, wz]
        p0 = self.plant.trunk_pos
        min_up = 1.0
        t_end = self.plant.sim_time + seconds
        while self.plant.sim_time < t_end and self.ctl.state == WALK:
            self.tick()
            min_up = min(min_up, self.plant.upright())
        p1 = self.plant.trunk_pos
        return {"dx": float(p1[0] - p0[0]), "dy": float(p1[1] - p0[1]), "min_upright": min_up,
                "state": self.ctl.state, "sim_s": seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--vx", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--echo", action="store_true")
    a = ap.parse_args()
    r = SimRobot(hz=a.hz, echo=a.echo, quiet=False)
    print("read mode:", r.mode)
    r.stand_up()
    res = r.walk(a.vx, a.seconds)
    print(res)
    r.key(term.KEY_SPACE)
    r.run_until(lambda: r.ctl.state == STAND, 5)
    print("final state", r.ctl.state, "upright", round(r.plant.upright(), 3))


if __name__ == "__main__":
    main()
