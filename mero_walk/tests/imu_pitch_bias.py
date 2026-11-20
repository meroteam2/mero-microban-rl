"""True trunk pitch vs the runtime's IMU (Mahony) pitch while standing and walking, in the MuJoCo plant.

Runs the real runtime stack (mero.sim.harness.SimRobot) with the MERO 5 V setup
(30 Hz, kp_stand 620, kp_walk 190, motor supply 4.9 V) and prints, per phase,
the mean true pitch (from the simulated trunk orientation) and the mean pitch
the IMU filter reports (projected gravity), plus how often the accelerometer
was ignored (|acc| outside 0.5..1.5 g).

  python tests/imu_pitch_bias.py [--policy models/mero_walk_dr.onnx --gyro-frame body]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mero import term  # noqa: E402
from mero.battery import BatteryMonitor  # noqa: E402
from mero.controller import STAND, WALK  # noqa: E402
from mero.sim.harness import SimRobot  # noqa: E402


def true_pitch(r: SimRobot) -> float:
  w, x, y, z = r.plant.trunk_quat
  # trunk +z (up) axis in the world; pitch forward = +x tilt of the up axis
  up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
  return math.degrees(math.atan2(up[0], up[2]))


def imu_pitch(r: SimRobot) -> float:
  g = r.imu.latest().proj_grav
  return math.degrees(math.atan2(g[0], -g[2]))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default=None)
  ap.add_argument("--gyro-frame", default="imu")
  ap.add_argument("--hz", type=float, default=30.0)
  ap.add_argument("--vin", type=float, default=4.9)
  ap.add_argument("--kp-walk", type=int, default=190)
  ap.add_argument("--kp-stand", type=int, default=620)
  ap.add_argument("--seconds", type=float, default=8.0)
  a = ap.parse_args()

  for vx in (0.0, 0.1, 0.2, 0.3):
    r = SimRobot(hz=a.hz, vin=a.vin, gyro_frame=a.gyro_frame, policy_path=a.policy)
    r.cfg["control"]["kp_walk"] = a.kp_walk
    r.cfg["control"]["kp_stand"] = a.kp_stand
    # Motors on the 5 V regulator: judge the (motor-measured) supply with 1-cell thresholds.
    r.cfg["battery"].update({"source": "motor", "cells": 1, "warn_v": 4.75, "low_v": 4.6, "critical_v": 4.4})
    r.mon = r.ctl.mon = BatteryMonitor(r.cfg)
    n_bad = n_all = 0
    orig = r.imu.dev.read

    def read():
      nonlocal n_bad, n_all
      acc, gyr = orig()
      n_all += 1
      n_bad += not (0.5 < np.linalg.norm(acc) < 1.5)
      return acc, gyr

    r.imu.dev.read = read
    r.stand_up()
    r.tick(int(a.hz * 1.5))
    st = [(true_pitch(r), imu_pitch(r))]
    for _ in range(int(a.hz)):
      r.tick()
      st.append((true_pitch(r), imu_pitch(r)))
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [vx, 0.0, 0.0]
    n_bad = n_all = 0
    wk = []
    p0 = r.plant.trunk_pos
    for k in range(int(a.seconds * a.hz)):
      r.tick()
      if r.ctl.state != WALK:
        break
      if k > a.hz * 2:
        wk.append((true_pitch(r), imu_pitch(r)))
    st, wk = np.array(st), np.array(wk) if wk else np.full((1, 2), np.nan)
    dx = r.plant.trunk_pos[0] - p0[0]
    print(f"vx {vx:.1f}: stand true {st[:, 0].mean():5.1f} imu {st[:, 1].mean():5.1f} | "
          f"walk true {wk[:, 0].mean():5.1f} imu {wk[:, 1].mean():5.1f} (imu-true {np.nanmean(wk[:, 1] - wk[:, 0]):+5.1f}) | "
          f"acc ignored {100 * n_bad / max(n_all, 1):4.1f}% | state {r.ctl.state} dx {dx:+.2f} m")


if __name__ == "__main__":
  main()
