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


def test_mixed_firmware_uses_indirect_with_shifted_slots():
    """MERO chain: V52 and V53 motors mixed (Indirect Data 1 at 208 vs 224)."""
    ids = [MOTOR_ID[n] for n in JOINTS]
    ser, motors = make_chain(ids, firmware=53)
    for mid in ids[::2]:
        motors[mid].ct[dxl.ADDR_FIRMWARE] = 52
    bus = MotorBus(dxl.Protocol2(ser=ser), Calibration())
    assert bus.setup("auto") == "fast_indirect"
    motors[24].set_present(1000, 5)     # V53
    motors[11].set_present(3100, -7)    # V52
    pos, vel = bus.read_raw()
    i24, i11 = JOINTS.index("right_knee"), JOINTS.index("left_hip_yaw")
    assert (pos[i24], vel[i24]) == (1000, 5)
    assert (pos[i11], vel[i11]) == (3100, -7)


def test_swap_leg_ids_through_temp_ids():
    import struct
    from mero import dxl as _dxl
    from mero.app_tools import LEG_SWAP, swap_ids
    from mero.joints import MOTOR_ID
    from mero.sim.fake_dxl import make_chain
    ser, motors = make_chain(sorted(MOTOR_ID.values()))

    def rekey(addr, _written):            # a real motor answers to its new ID right away
        if addr == _dxl.ADDR_ID:
            for m in list(ser.motors.values()):
                if m.ct[_dxl.ADDR_ID] != m.id:
                    del ser.motors[m.id]
                    m.id = m.ct[_dxl.ADDR_ID]
                    ser.motors[m.id] = m
    ser.on_write = rekey
    for i, m in ser.motors.items():           # fingerprint each physical motor by its position
        struct.pack_into("<i", m.ct, _dxl.ADDR_PRESENT_POSITION, 1000 + i)
    p = _dxl.Protocol2(ser=ser)
    swap_ids(p, LEG_SWAP, log=lambda *_: None)
    assert sorted(ser.motors) == sorted(MOTOR_ID.values())          # same ID set, no temp IDs left
    for a, b in LEG_SWAP:
        assert struct.unpack_from("<i", ser.motors[a].ct, _dxl.ADDR_PRESENT_POSITION)[0] == 1000 + b
        assert struct.unpack_from("<i", ser.motors[b].ct, _dxl.ADDR_PRESENT_POSITION)[0] == 1000 + a
    for i in (31, 41, 51):                                          # arms/head untouched
        assert struct.unpack_from("<i", ser.motors[i].ct, _dxl.ADDR_PRESENT_POSITION)[0] == 1000 + i


def test_swap_leg_ids_refuses_when_a_motor_is_missing():
    import pytest
    from mero import dxl as _dxl
    from mero.app_tools import LEG_SWAP, swap_ids
    from mero.joints import MOTOR_ID
    from mero.sim.fake_dxl import make_chain
    ser, _ = make_chain([i for i in MOTOR_ID.values() if i != 24])
    with pytest.raises(RuntimeError, match="24"):
        swap_ids(_dxl.Protocol2(ser=ser), LEG_SWAP, log=lambda *_: None)
    assert 11 in ser.motors and all(i < 100 for i in ser.motors)   # nothing was renamed


def _mt_bus(present: int, mode: int = 4):
    """right_shoulder_pitch horn ~180° off: zero -180°, sign -1, reference pose at raw ~0."""
    import math, struct
    from mero import dxl as _dxl
    from mero.bus import MotorBus
    from mero.calibration import Calibration
    from mero.joints import MOTOR_ID
    from mero.sim.fake_dxl import make_chain
    ser, motors = make_chain(sorted(MOTOR_ID.values()))
    for m in motors.values():
        struct.pack_into("<i", m.ct, _dxl.ADDR_PRESENT_POSITION, 2048)
    m41 = motors[41]
    m41.ct[_dxl.ADDR_OPERATING_MODE] = mode
    struct.pack_into("<i", m41.ct, _dxl.ADDR_PRESENT_POSITION, present)
    cal = Calibration()
    cal.joints["right_shoulder_pitch"].zero = -math.pi
    bus = MotorBus(_dxl.Protocol2(ser=ser), cal, multiturn=["right_shoulder_pitch"])
    return bus, m41


def _goal(m):
    import struct
    from mero import dxl as _dxl
    return struct.unpack_from("<i", m.ct, _dxl.ADDR_GOAL_POSITION)[0]


def test_multiturn_joint_goal_stays_on_the_current_turn():
    import math
    # just below the wrap (multi-turn count negative) and the same pose after the motor re-wrapped
    for present, want_goal in ((-60, -114), (4036, 4096 - 114)):
        bus, m41 = _mt_bus(present)
        for mode in ("auto", "sync"):
            bus.setup(mode)
            q, _, _ = bus.read_state()
            assert abs(math.degrees(q["right_shoulder_pitch"]) - 5.27) < 0.1, (present, mode)
        bus.write_goal({"right_shoulder_pitch": math.radians(10)})   # stand pose
        assert _goal(m41) == want_goal, present                      # unclamped, a few ticks away
        assert abs(_goal(m41) - present) < 100                       # no 360° trip


def test_multiturn_turn_is_frozen_while_torque_is_on():
    import math, struct
    from mero import dxl as _dxl
    bus, m41 = _mt_bus(-60)
    bus.setup("sync")
    bus.torque(True, ["right_shoulder_pitch"])
    struct.pack_into("<i", m41.ct, _dxl.ADDR_PRESENT_POSITION, -60 - 2200)   # arm swung >180° back
    q, _, _ = bus.read_state()
    assert math.degrees(q["right_shoulder_pitch"]) > 180                    # reported honestly
    bus.write_goal({"right_shoulder_pitch": 0.0})
    assert _goal(m41) == 0                                                   # same turn as before
    bus.torque_off_all_robust()
    q, _, _ = bus.read_state()                                               # torque off: re-pick turn
    assert -math.pi <= q["right_shoulder_pitch"] <= math.pi


def test_multiturn_joint_in_position_mode_is_refused():
    import pytest
    from mero import dxl as _dxl
    bus, _ = _mt_bus(-60, mode=3)
    with pytest.raises(_dxl.DxlError, match="check --fix"):
        bus.setup("sync")


def test_other_joints_still_clamped_to_one_turn():
    import math
    from mero.joints import MOTOR_ID
    bus, _ = _mt_bus(-60)
    bus.setup("sync")
    bus.cal.joints["left_shoulder_pitch"].zero = -math.pi
    bus.write_goal({"left_shoulder_pitch": math.radians(-10)})
    m = bus.p.ser.motors[MOTOR_ID["left_shoulder_pitch"]]
    assert _goal(m) == 0


def test_deadline_waits_for_queued_tx():
    """A read sent right after an unanswered sync write goes out only once the write is on the
    wire; its response deadline must count from then (calib stand hold: 'timeout ... have 142')."""
    from mero import dxl as _dxl

    class QueuedSerial:                     # write() returns at once, like pyserial
        def reset_input_buffer(self): pass
        def write(self, b): return len(b)

    p = _dxl.Protocol2(ser=QueuedSerial(), baud=115200)
    p._send(bytes(109))                     # 19 goals ~ 9.5 ms on the wire
    p._send(bytes(30))                      # read request queued behind it
    d = p._deadline(30, 160)
    import time
    budget = d - time.perf_counter()
    alone = (30 + 160) * p.byte_time * 1.25 + p.margin_s
    assert budget > alone + 109 * p.byte_time * 0.9
