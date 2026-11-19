"""Byte-level fake Dynamixel chain for tests (no hardware needed).

FakeSerial behaves like a pyserial port: write() takes instruction packets, read()
returns status packets. Each FakeMotor holds a 256+ byte control table. A `plant`
object can supply positions/velocities and receive goal positions.
"""
from __future__ import annotations

import struct
import time

from .. import dxl


class FakeMotor:
    def __init__(self, dxl_id: int, model: int = 1200, firmware: int = 53):
        self.id = dxl_id
        self.ct = bytearray(700)
        struct.pack_into("<H", self.ct, dxl.ADDR_MODEL, model)
        self.ct[dxl.ADDR_FIRMWARE] = firmware
        self.ct[dxl.ADDR_ID] = dxl_id
        self.ct[dxl.ADDR_BAUD] = 2
        self.ct[dxl.ADDR_RETURN_DELAY] = 0
        self.ct[dxl.ADDR_OPERATING_MODE] = 3
        struct.pack_into("<H", self.ct, dxl.ADDR_MAX_VOLTAGE, 70)
        struct.pack_into("<H", self.ct, dxl.ADDR_MIN_VOLTAGE, 35)
        struct.pack_into("<H", self.ct, dxl.ADDR_PWM_LIMIT, 885)
        struct.pack_into("<H", self.ct, dxl.ADDR_CURRENT_LIMIT, 1750)
        struct.pack_into("<I", self.ct, dxl.ADDR_MAX_POSITION, 4095)
        struct.pack_into("<I", self.ct, dxl.ADDR_MIN_POSITION, 0)
        self.ct[dxl.ADDR_PWM_SLOPE] = 255
        self.ct[dxl.ADDR_SHUTDOWN] = 52
        self.ct[dxl.ADDR_STATUS_RETURN] = 2
        struct.pack_into("<H", self.ct, dxl.ADDR_POSITION_P, 400)
        struct.pack_into("<i", self.ct, dxl.ADDR_PRESENT_POSITION, 2048)
        struct.pack_into("<i", self.ct, dxl.ADDR_GOAL_POSITION, 2048)
        struct.pack_into("<H", self.ct, dxl.ADDR_PRESENT_VOLTAGE, 78)
        self.ct[dxl.ADDR_PRESENT_TEMPERATURE] = 30
        for k in range(20):  # default indirect addresses
            struct.pack_into("<H", self.ct, dxl.ADDR_INDIRECT_ADDRESS_1 + 2 * k, 224 + k)

    @property
    def firmware(self) -> int:
        return self.ct[dxl.ADDR_FIRMWARE]

    def _ind_data_base(self) -> int:
        return dxl.ADDR_INDIRECT_DATA_1_NEW if self.firmware >= 53 else dxl.ADDR_INDIRECT_DATA_1_OLD

    def _resolve(self, addr: int) -> int:
        base = self._ind_data_base()
        if base <= addr < base + 20:
            k = addr - base
            return struct.unpack_from("<H", self.ct, dxl.ADDR_INDIRECT_ADDRESS_1 + 2 * k)[0]
        return addr

    def read(self, addr: int, n: int) -> bytes:
        return bytes(self.ct[self._resolve(a)] for a in range(addr, addr + n))

    def write(self, addr: int, data: bytes) -> None:
        torque = self.ct[dxl.ADDR_TORQUE_ENABLE]
        for k, b in enumerate(data):
            a = self._resolve(addr + k)
            if a < 64 and torque:
                return  # EEPROM is locked while torque is on
            self.ct[a] = b

    def set_present(self, pos: int, vel: int) -> None:
        struct.pack_into("<i", self.ct, dxl.ADDR_PRESENT_POSITION, int(pos))
        struct.pack_into("<i", self.ct, dxl.ADDR_PRESENT_VELOCITY, int(vel))

    @property
    def goal(self) -> int:
        return struct.unpack_from("<i", self.ct, dxl.ADDR_GOAL_POSITION)[0]

    @property
    def torque(self) -> bool:
        return bool(self.ct[dxl.ADDR_TORQUE_ENABLE])

    @property
    def kp(self) -> int:
        return struct.unpack_from("<H", self.ct, dxl.ADDR_POSITION_P)[0]


def _status(dxl_id: int, data: bytes = b"", err: int = 0) -> bytes:
    return dxl.build_packet(dxl_id, dxl.INST_STATUS, bytes([err]) + data)


class FakeSerial:
    """pyserial-like port connected to FakeMotors.

    echo: also return our own instruction bytes (single-wire half duplex).
    baud: if given, read() sleeps for the real wire time of tx+rx bytes.
    """

    def __init__(self, motors: list[FakeMotor], echo: bool = False, baud: int | None = None,
                 fast_sync: bool = True):
        self.motors = {m.id: m for m in motors}
        self.echo = echo
        self.baud = baud
        self.fast_sync = fast_sync
        self._rx = bytearray()
        self.timeout = 0.004
        self.on_write = None          # callback(addr, {id: bytes}) after any write/sync write
        self.before_read = None       # callback() before present values are served
        self._wire_s = 0.0
        self.closed = False

    # pyserial API
    def reset_input_buffer(self) -> None:
        self._rx.clear()

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def close(self) -> None:
        self.closed = True

    def read(self, n: int) -> bytes:
        if self.baud and self._wire_s > 0:
            time.sleep(self._wire_s)
            self._wire_s = 0.0
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def write(self, data: bytes) -> int:
        if self.echo:
            self._rx += data
        resp = self._handle(bytes(data))
        self._rx += resp
        if self.baud:
            self._wire_s += (len(data) + len(resp)) * 10.0 / self.baud
        return len(data)

    # protocol
    def _handle(self, pkt: bytes) -> bytes:
        if len(pkt) < 10 or pkt[:4] != dxl.HEADER:
            return b""
        if dxl.crc16(pkt[:-2]) != (pkt[-2] | (pkt[-1] << 8)):
            return b""
        pid, inst = pkt[4], pkt[7]
        params = dxl.unstuff(pkt[8:-2])
        targets = list(self.motors.values()) if pid == dxl.BROADCAST else (
            [self.motors[pid]] if pid in self.motors else [])
        out = bytearray()
        if inst == dxl.INST_PING:
            for m in targets:
                out += _status(m.id, struct.pack("<HB", struct.unpack_from("<H", m.ct, 0)[0], m.firmware))
        elif inst == dxl.INST_READ and pid != dxl.BROADCAST:
            addr, n = struct.unpack("<HH", params[:4])
            if self.before_read:
                self.before_read()
            for m in targets:
                out += _status(m.id, m.read(addr, n))
        elif inst == dxl.INST_WRITE and pid != dxl.BROADCAST:
            addr = struct.unpack("<H", params[:2])[0]
            for m in targets:
                m.write(addr, params[2:])
                if m.ct[dxl.ADDR_STATUS_RETURN] == 2:
                    out += _status(m.id)
            if self.on_write:
                self.on_write(addr, {m.id: params[2:] for m in targets})
        elif inst == dxl.INST_SYNC_WRITE:
            addr, n = struct.unpack("<HH", params[:4])
            body = params[4:]
            written = {}
            for k in range(0, len(body), n + 1):
                mid = body[k]
                if mid in self.motors:
                    self.motors[mid].write(addr, body[k + 1:k + 1 + n])
                    written[mid] = body[k + 1:k + 1 + n]
            if self.on_write:
                self.on_write(addr, written)
        elif inst == dxl.INST_SYNC_READ:
            addr, n = struct.unpack("<HH", params[:4])
            if self.before_read:
                self.before_read()
            for mid in params[4:]:
                if mid in self.motors:
                    out += _status(mid, self.motors[mid].read(addr, n))
        elif inst == dxl.INST_FAST_SYNC_READ and self.fast_sync:
            addr, n = struct.unpack("<HH", params[:4])
            if self.before_read:
                self.before_read()
            ids = [i for i in params[4:] if i in self.motors]
            if len(ids) != len(params[4:]):
                return b""  # chain broken: a real bus would time out
            body = bytearray([dxl.INST_STATUS])
            for k, mid in enumerate(ids):
                part = bytes([0, mid]) + self.motors[mid].read(addr, n)
                body += part
                if k != len(ids) - 1:
                    c = dxl.crc16(part)
                    body += bytes([c & 0xFF, c >> 8])
            length = len(body) + 2
            pk = dxl.HEADER + bytes([dxl.BROADCAST, length & 0xFF, length >> 8]) + body
            c = dxl.crc16(pk)
            out += pk + bytes([c & 0xFF, c >> 8])
        return bytes(out)


def make_chain(ids, firmware: int = 53, **kw) -> tuple[FakeSerial, dict[int, FakeMotor]]:
    motors = [FakeMotor(i, firmware=firmware) for i in ids]
    return FakeSerial(motors, **kw), {m.id: m for m in motors}
