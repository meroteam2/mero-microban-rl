"""Protocol tests: e-manual example packets + fake chain round trips."""
import struct
import pytest
from mero import dxl
from mero.sim.fake_dxl import make_chain
from mero.joints import JOINTS, MOTOR_ID
from mero.bus import MotorBus
from mero.calibration import Calibration

H = bytes.fromhex


def test_manual_examples():
    assert dxl.build_packet(1, dxl.INST_PING) == H("FFFFFD00010300 01 194E".replace(" ", ""))
    assert dxl.build_packet(0xFE, dxl.INST_SYNC_READ, H("84000400") + bytes([1, 2])) == H("FFFFFD00FE0900828400040001 02CEFA".replace(" ", ""))
    sw = dxl.build_packet(0xFE, dxl.INST_SYNC_WRITE, H("74000400") + H("0196000000") + H("02AA000000"))
    assert sw == H("FFFFFD00FE1100837400040001960000000 2AA00000082 87".replace(" ", ""))
    fsr = dxl.build_packet(0xFE, dxl.INST_FAST_SYNC_READ, H("84000400") + bytes([3, 7, 4]))
    assert fsr == H("FFFFFD00FE0A008A840004000307 0420F2".replace(" ", ""))


class _Canned:
    def __init__(self, data): self.data = bytearray(data); self.timeout = 0.004
    def reset_input_buffer(self): pass
    def write(self, b): return len(b)
    def read(self, n):
        out = bytes(self.data[:n]); del self.data[:n]; return out


def test_parse_manual_fast_sync_status():
    resp = H("FFFFFD00FE190055000 3A600000084080007 1F08000016CA0004FF030000D19E".replace(" ", ""))
    p = dxl.Protocol2(ser=_Canned(resp))
    out = p.fast_sync_read(0x84, 4, [3, 7, 4])
    assert struct.unpack("<i", out[3])[0] == 0xA6
    assert struct.unpack("<i", out[7])[0] == 0x81F
    assert struct.unpack("<i", out[4])[0] == 0x3FF


def test_stuffing_roundtrip():
    body = bytes([0x03, 0xFF, 0xFF, 0xFD, 0x01])
    assert dxl.unstuff(dxl.stuff(body)) == body
    assert dxl.stuff(body) == bytes([0x03, 0xFF, 0xFF, 0xFD, 0xFD, 0x01])


@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("fw", [46, 53])
def test_chain_roundtrip(echo, fw):
    ids = [MOTOR_ID[n] for n in JOINTS]
    ser, motors = make_chain(ids, firmware=fw, echo=echo)
    p = dxl.Protocol2(ser=ser)
    assert p.ping(11) == (1200, fw)
    assert p.ping(99) is None
    assert set(p.broadcast_ping(0.02)) == set(ids)
    motors[14].set_present(3000, -25)
    bus = MotorBus(p, Calibration())
    mode = bus.setup("auto")
    assert mode == "fast_indirect"
    q, dq, raw = bus.read_state()
    assert raw[JOINTS.index("left_knee")] == 3000
    c = bus.cal.joints["left_knee"]
    assert abs(q["left_knee"] - c.to_sim(3000)) < 1e-9
    assert dq["left_knee"] < 0
    bus.write_goal({"left_knee": 0.5})
    assert motors[14].goal == c.to_raw(0.5)
    assert abs(bus.read_voltage("head") - 7.8) < 1e-9
    bus.torque(True)
    assert all(m.torque for m in motors.values())


def test_fallback_without_fast_sync():
    ids = [MOTOR_ID[n] for n in JOINTS]
    ser, motors = make_chain(ids, firmware=45)
    bus = MotorBus(dxl.Protocol2(ser=ser), Calibration())
    assert bus.setup("auto") == "sync"
    ser2, _ = make_chain(ids, firmware=53)
    ser2.fast_sync = False
    bus2 = MotorBus(dxl.Protocol2(ser=ser2), Calibration())
    with pytest.raises(dxl.DxlError):
        bus2.p.fast_sync_read(132, 4, ids)


def test_mixed_firmware_uses_direct_fast():
    ids = [MOTOR_ID[n] for n in JOINTS]
    ser, motors = make_chain(ids, firmware=53)
    motors[11].ct[dxl.ADDR_FIRMWARE] = 50
    bus = MotorBus(dxl.Protocol2(ser=ser), Calibration())
    assert bus.setup("auto") == "fast"
    motors[24].set_present(1000, 5)
    _, _, raw = bus.read_state()
    assert raw[JOINTS.index("right_knee")] == 1000
