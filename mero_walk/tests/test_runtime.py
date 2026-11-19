"""Closed-loop runtime tests on the MuJoCo plant through the byte-level fake bus."""
import math
import numpy as np
import pytest
from mero import term, battery as bat
from mero.controller import STAND, WALK, IDLE, FALLEN, STOPPING
from mero.sim.harness import SimRobot


def test_walk_forward_and_stop():
    r = SimRobot(hz=40)
    r.stand_up()
    res = r.walk(0.15, 8.0)
    assert res["state"] == WALK and res["min_upright"] > 0.9
    assert math.hypot(res["dx"], res["dy"]) > 0.6
    r.key(term.KEY_SPACE)
    assert r.run_until(lambda: r.ctl.state == STAND, 5.0)
    assert r.plant.upright() > 0.95


def test_battery_levels_stop_walk_then_torque_off():
    r = SimRobot(hz=40)
    r.cfg["battery"]["read_period_s"] = 0.2
    r.mon.tau = 0.5
    r.cfg["battery"]["critical_countdown_s"] = 2.0
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [0.1, 0, 0]
    r.run_until(lambda: False, 2.0)
    r.plant.vin = 6.95                        # WARN: keep walking
    r.run_until(lambda: r.mon.status.level == bat.WARN, 5.0)
    assert r.ctl.state == WALK
    r.plant.vin = 6.7                         # LOW: stop and refuse
    assert r.run_until(lambda: r.ctl.state == STAND, 8.0), r.events
    r.key(term.KEY_SPACE)
    assert r.ctl.state == STAND and "걷기 금지" in r.events[-1]
    r.plant.vin = 6.3                         # CRITICAL: countdown then torque off
    assert r.run_until(lambda: r.ctl.state == IDLE, 8.0), r.events
    assert not any(m.torque for m in r.plant.motors.values())
    assert any(m.ct[65] for m in r.plant.motors.values())   # red LEDs on
    r.key("t")
    assert r.ctl.state == IDLE                # refuses torque on


def test_fall_detection_cuts_torque():
    r = SimRobot(hz=40)
    r.stand_up()
    r.key(term.KEY_SPACE)
    trunk = r.plant.m.body("trunk").id
    for _ in range(40):
        r.plant.d.xfrc_applied[trunk, 0] = 25.0     # strong shove forward
        r.tick()
        if r.ctl.state != WALK:
            break
    r.plant.d.xfrc_applied[trunk, :] = 0
    r.run_until(lambda: r.ctl.state == FALLEN, 3.0)
    assert r.ctl.state == FALLEN
    assert not any(m.torque for m in r.plant.motors.values())


def test_connection_lost_while_walking_stops_then_quits():
    r = SimRobot(hz=40)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [0.15, 0, 0]
    r.run_until(lambda: False, 2.0)
    r.key(term.KEY_EOF)
    assert r.ctl.state == STOPPING
    assert r.run_until(lambda: r.ctl.quit, 6.0)
    assert r.plant.upright() > 0.95


def test_emergency_stop_and_torque_guard():
    r = SimRobot(hz=40)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.key("t")                                  # refused while walking
    assert r.ctl.state == WALK
    r.key("k")
    assert r.ctl.state == IDLE
    assert not any(m.torque for m in r.plant.motors.values())


def test_kp_switches_between_stand_and_walk():
    r = SimRobot(hz=40)
    r.stand_up()
    assert {m.kp for m in r.plant.motors.values()} == {400}
    r.key(term.KEY_SPACE)
    r.tick()
    assert {m.kp for m in r.plant.motors.values()} == {125}
    r.key(term.KEY_SPACE)
    r.run_until(lambda: r.ctl.state == STAND, 5.0)
    assert {m.kp for m in r.plant.motors.values()} == {400}


def test_refuses_torque_when_joint_out_of_range():
    from mero.calibration import Calibration
    import math as _m
    true = Calibration()
    true.joints["left_hip_roll"].zero = _m.radians(60)   # badly assembled / wrong zero
    r = SimRobot(hz=40, true_cal=true)
    r.tick(3)
    r.key("t")
    assert r.ctl.state == IDLE
    assert "범위 밖" in r.events[-1]
    assert not any(m.torque for m in r.plant.motors.values())


def test_imu_dropout_while_walking_goes_to_stand():
    import time as _t
    r = SimRobot(hz=40)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [0.1, 0, 0]
    r.run_until(lambda: False, 1.5)
    r.imu_frozen = True
    _t.sleep(0.3)                      # sample timestamps are wall-clock
    r.tick()
    assert r.ctl.state == STOPPING and r.ctl.stop_phase == 1
    assert r.run_until(lambda: r.ctl.state == STAND, 3.0)


def test_walk_stop_walk_again():
    r = SimRobot(hz=30)
    r.stand_up()
    for _ in range(2):
        res = r.walk(0.1, 3.0)
        assert res["state"] == WALK and r.ctl.stop_phase == 0
        r.key(term.KEY_SPACE)
        assert r.run_until(lambda: r.ctl.state == STAND, 5.0)


def test_walk_log_from_start_to_stand(tmp_path):
    import csv, json
    from mero.walklog import WalkLog
    r = SimRobot(hz=40)
    r.ctl.log = WalkLog(tmp_path)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [0.1, 0, 0]
    r.run_until(lambda: False, 2.0)
    assert r.ctl.log.active
    r.key(term.KEY_SPACE)
    assert r.run_until(lambda: r.ctl.state == STAND, 5.0)
    assert not r.ctl.log.active                                   # closed when standing again
    files = sorted(tmp_path.glob("walk_*.csv"))
    assert len(files) == 1
    rows = list(csv.DictReader(open(files[0], encoding="utf-8")))
    assert len(rows) >= 2.0 * 40 * 0.9
    assert {r_["state"] for r_ in rows} >= {"WALK", "STOPPING"}
    pol = r.ctl.policy
    walk = [r_ for r_ in rows if r_["state"] == "WALK"]
    assert all(r_[f"obs_{pol.obs_size - 1}"] != "" for r_ in walk[1:])          # policy obs logged
    assert all(r_[f"act_{pol.order[0]}"] != "" and r_[f"tgt_{pol.order[0]}"] != "" for r_ in walk[1:])
    assert all(r_["grav_z"] != "" and r_["q_left_knee"] != "" for r_ in rows)
    assert any("걷기 시작" in r_["event"] for r_ in rows)
    assert float(walk[-1]["cmd_vx"]) > 0.05
    meta = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["obs_size"] == pol.obs_size and meta["order"] == pol.order
    assert any("걷기 로그 저장" in e for e in r.events)


def test_walk_log_closed_on_torque_off(tmp_path):
    import csv
    from mero.walklog import WalkLog
    r = SimRobot(hz=40)
    r.ctl.log = WalkLog(tmp_path)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.run_until(lambda: False, 0.5)
    r.key("k")                                                     # emergency stop while walking
    assert not r.ctl.log.active
    rows = list(csv.DictReader(open(next(tmp_path.glob("walk_*.csv")), encoding="utf-8")))
    assert rows and rows[-1]["event"] != ""                        # last row carries the stop message
