"""INA219 battery sensor (pack side of the 5 V regulator) and its use in the runtime."""
import json
from pathlib import Path

import pytest

from mero import battery as bat, config, term
from mero.controller import IDLE, STAND, WALK
from mero.sim.harness import SimRobot


class FakeSMBus:
    """INA219 bus-voltage register over SMBus (words arrive byte-swapped)."""

    def __init__(self, volts: float, addr: int = 0x40):
        self.volts, self.addr = volts, addr

    def read_word_data(self, addr: int, reg: int) -> int:
        if addr != self.addr:
            raise OSError(121, "Remote I/O error")
        assert reg == bat.INA219.REG_BUS_V
        w = (int(round(self.volts / 0.004)) << 3) | 0b10    # CNVR set, like the chip
        return ((w & 0xFF) << 8) | (w >> 8)


def _cfg(**battery):
    cfg = config.load("/nonexistent")
    cfg["battery"].update({"source": "ina219", **battery})
    return cfg


def test_ina219_conversion_and_scale():
    assert bat.INA219(smbus=FakeSMBus(7.62)).read() == pytest.approx(7.62, abs=0.004)
    assert bat.INA219(smbus=FakeSMBus(3.81), scale=2.0).read() == pytest.approx(7.62, abs=0.01)


def test_open_sensor_motor_source_and_errors():
    assert bat.open_sensor(config.load("/nonexistent")) is None          # default: motor voltage
    with pytest.raises(OSError):
        bat.open_sensor(_cfg(), smbus=FakeSMBus(7.6, addr=0x41))          # wrong address
    with pytest.raises(RuntimeError):
        bat.open_sensor(_cfg(), smbus=FakeSMBus(0.0))                     # answers but nothing wired
    with pytest.raises(ValueError):
        bat.open_sensor(_cfg(source="ads9999"))


def test_missing_sensor_reader_keeps_failing_and_warns():
    msgs = []
    read = bat.voltage_reader(_cfg(), emit=msgs.append, smbus=FakeSMBus(7.6, addr=0x41))
    assert read is not None and "배터리 보호" in msgs[0]
    with pytest.raises(RuntimeError):
        read()


def test_example_config_is_valid():
    p = Path(__file__).resolve().parent.parent / "config.5v.example.json"
    cfg = config.load(p)
    assert cfg["battery"]["source"] == "ina219" and cfg["battery"]["sensor"]["addr"] == 0x40
    assert cfg["control"]["kp_walk"] > config.DEFAULTS["control"]["kp_walk"]
    json.loads(p.read_text(encoding="utf-8"))


def test_runtime_uses_sensor_not_regulated_motor_voltage():
    """Motors see a steady 4.9 V; only the pack voltage from the sensor drives the levels."""
    r = SimRobot(hz=40, vin=4.9 * 1.55)   # plant strength as with the raised 5 V gains
    r.plant.vin = 7.6
    r.cfg["battery"]["read_period_s"] = 0.2
    r.cfg["battery"]["critical_countdown_s"] = 2.0
    r.mon.tau = 0.5
    fake = FakeSMBus(7.8)
    r.ctl.vread = bat.voltage_reader(_cfg(), emit=lambda m: None, smbus=fake)
    r.stand_up()
    r.key(term.KEY_SPACE)
    r.ctl.cmd_des[:] = [0.1, 0, 0]
    r.run_until(lambda: False, 1.5)
    assert r.ctl.state == WALK and r.mon.status.level == bat.OK
    assert r.mon.status.filtered_v == pytest.approx(7.8, abs=0.05)
    fake.volts = 6.7                                  # pack LOW while the motors still get 4.9 V
    assert r.run_until(lambda: r.ctl.state == STAND, 8.0), r.events
    fake.volts = 6.2                                  # CRITICAL: countdown then torque off
    assert r.run_until(lambda: r.ctl.state == IDLE, 8.0), r.events


def test_sensor_failure_mid_run_warns_without_stopping():
    r = SimRobot(hz=40)
    r.cfg["battery"]["read_period_s"] = 0.2
    fake = FakeSMBus(7.6)
    r.ctl.vread = bat.voltage_reader(_cfg(), emit=lambda m: None, smbus=fake)
    r.stand_up()
    fake.addr = 0x41                                  # wire came loose
    r.run_until(lambda: False, 1.5)
    assert r.ctl.state == STAND
    assert any("배터리 전압을 읽을 수 없습니다" in m for m in r.events)
    assert r.mon.line() == "배터리 측정 불가"
    fake.addr = 0x40
    r.run_until(lambda: False, 0.5)
    assert r.mon.status.error == "" and r.mon.status.level == bat.OK
