"""Minimal Dynamixel Protocol 2.0 driver (pyserial), tuned for a slow 115200 bus.

Only what the robot needs: ping, read, write, sync write, sync read and fast sync
read. Packets echoed back by a single-wire half-duplex circuit are ignored.
Reference: https://emanual.robotis.com/docs/en/dxl/protocol2/
"""
from __future__ import annotations

import struct
import time

HEADER = b"\xff\xff\xfd\x00"
BROADCAST = 0xFE

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_REBOOT = 0x08
INST_STATUS = 0x55
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
INST_FAST_SYNC_READ = 0x8A

# XL330 control table (addresses used here)
ADDR_MODEL = 0
ADDR_FIRMWARE = 6
ADDR_ID = 7
ADDR_BAUD = 8
ADDR_RETURN_DELAY = 9
ADDR_DRIVE_MODE = 10
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_MAX_VOLTAGE = 32
ADDR_MIN_VOLTAGE = 34
ADDR_PWM_LIMIT = 36
ADDR_CURRENT_LIMIT = 38
ADDR_MAX_POSITION = 48
ADDR_MIN_POSITION = 52
ADDR_PWM_SLOPE = 62
ADDR_SHUTDOWN = 63
ADDR_TORQUE_ENABLE = 64
ADDR_LED = 65
ADDR_STATUS_RETURN = 68
ADDR_HW_ERROR = 70
ADDR_POSITION_P = 84
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_INDIRECT_ADDRESS_1 = 168
ADDR_INDIRECT_DATA_1_OLD = 208   # firmware < V53
ADDR_INDIRECT_DATA_1_NEW = 224   # firmware >= V53

MODEL_NAMES = {1200: "XL330-M288", 1190: "XL330-M077", 1210: "XC330-M181", 1220: "XC330-M288",
               1230: "XC330-T181", 1240: "XC330-T288", 1060: "XL430-W250"}

BAUD_CODES = {0: 9600, 1: 57600, 2: 115200, 3: 1_000_000, 4: 2_000_000, 5: 3_000_000, 6: 4_000_000}


class DxlError(Exception):
    pass


class DxlTimeout(DxlError):
    pass


def _make_crc_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


_CRC_TABLE = _make_crc_table()


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _CRC_TABLE[((crc >> 8) ^ b) & 0xFF]) & 0xFFFF
    return crc


def stuff(body: bytes) -> bytes:
    """Insert 0xFD after every FF FF FD inside instruction+parameters."""
    if b"\xff\xff\xfd" not in body:
        return body
    out = bytearray()
    i = 0
    while i < len(body):
        if body[i:i + 3] == b"\xff\xff\xfd":
            out += b"\xff\xff\xfd\xfd"
            i += 3
        else:
            out.append(body[i])
            i += 1
    return bytes(out)


def unstuff(body: bytes) -> bytes:
    return body.replace(b"\xff\xff\xfd\xfd", b"\xff\xff\xfd")


def build_packet(dxl_id: int, inst: int, params: bytes = b"") -> bytes:
    body = stuff(bytes([inst]) + bytes(params))
    length = len(body) + 2
    pkt = HEADER + bytes([dxl_id, length & 0xFF, length >> 8]) + body
    c = crc16(pkt)
    return pkt + bytes([c & 0xFF, c >> 8])


def u16(v: int) -> bytes:
    return struct.pack("<H", v & 0xFFFF)


def u32(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)


def s32(v: int) -> bytes:
    return struct.pack("<i", v)


class Protocol2:
    """Blocking request/response transport on one serial port."""

    def __init__(self, port: str | None = None, baud: int = 115200, ser=None, margin_s: float = 0.008):
        if ser is None:
            import serial  # pyserial
            ser = serial.Serial(port, baud, timeout=0.004, write_timeout=0.2)
        self.ser = ser
        self.baud = baud
        self.byte_time = 10.0 / baud
        self.margin_s = margin_s
        self._buf = bytearray()
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.hw_alerts: dict[int, int] = {}   # id -> last status error byte with bit 7 set

    # ------------------------------------------------------------------ low level
    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def _send(self, pkt: bytes) -> None:
        self.ser.reset_input_buffer()
        self._buf.clear()
        self.ser.write(pkt)
        self.tx_bytes += len(pkt)

    def _deadline(self, tx: int, rx: int) -> float:
        return time.perf_counter() + (tx + rx) * self.byte_time * 1.25 + self.margin_s

    def _fill(self, n: int, deadline: float) -> None:
        while len(self._buf) < n:
            chunk = self.ser.read(n - len(self._buf))
            if chunk:
                self._buf += chunk
                self.rx_bytes += len(chunk)
            elif time.perf_counter() > deadline:
                raise DxlTimeout(f"timeout waiting for {n} bytes (have {len(self._buf)})")

    def _next_packet(self, deadline: float) -> tuple[int, int, bytes]:
        """Return (id, instruction, raw params) of the next CRC-valid packet."""
        while True:
            self._fill(7, deadline)
            idx = self._buf.find(HEADER)
            if idx < 0:
                del self._buf[:-3]
                continue
            if idx > 0:
                del self._buf[:idx]
                continue
            length = self._buf[5] | (self._buf[6] << 8)
            if length < 3 or length > 2048:
                del self._buf[:1]
                continue
            self._fill(7 + length, deadline)
            pkt = bytes(self._buf[:7 + length])
            if crc16(pkt[:-2]) != (pkt[-2] | (pkt[-1] << 8)):
                del self._buf[:1]
                continue
            del self._buf[:7 + length]
            return pkt[4], pkt[7], pkt[8:-2]

    def _status(self, deadline: float, expect_id: int | None = None) -> tuple[int, int, bytes]:
        """Next status packet (skips our own echoed instruction). Returns (id, error, data)."""
        while True:
            pid, inst, params = self._next_packet(deadline)
            if inst != INST_STATUS:
                continue  # echo of an instruction packet
            params = unstuff(params)
            if expect_id is not None and pid != expect_id:
                continue
            err = params[0] if params else 0
            if err & 0x80:
                self.hw_alerts[pid] = err
            if err & 0x7F:
                raise DxlError(f"ID {pid}: status error 0x{err:02x}")
            return pid, err, params[1:]

    # ------------------------------------------------------------------ instructions
    def ping(self, dxl_id: int) -> tuple[int, int] | None:
        pkt = build_packet(dxl_id, INST_PING)
        self._send(pkt)
        try:
            _, _, data = self._status(self._deadline(len(pkt), 14), dxl_id)
        except DxlTimeout:
            return None
        return data[0] | (data[1] << 8), data[2]

    def broadcast_ping(self, wait_s: float = 0.3) -> dict[int, tuple[int, int]]:
        pkt = build_packet(BROADCAST, INST_PING)
        self._send(pkt)
        found: dict[int, tuple[int, int]] = {}
        deadline = time.perf_counter() + wait_s
        while True:
            try:
                pid, _, data = self._status(deadline)
            except DxlTimeout:
                break
            except DxlError:
                continue
            if len(data) >= 3:
                found[pid] = (data[0] | (data[1] << 8), data[2])
        return found

    def read(self, dxl_id: int, addr: int, n: int) -> bytes:
        pkt = build_packet(dxl_id, INST_READ, u16(addr) + u16(n))
        self._send(pkt)
        _, _, data = self._status(self._deadline(len(pkt), 11 + n), dxl_id)
        if len(data) != n:
            raise DxlError(f"ID {dxl_id}: expected {n} bytes, got {len(data)}")
        return data

    def write(self, dxl_id: int, addr: int, data: bytes, ack: bool = True) -> None:
        pkt = build_packet(dxl_id, INST_WRITE, u16(addr) + bytes(data))
        self._send(pkt)
        if ack:
            try:
                self._status(self._deadline(len(pkt), 11), dxl_id)
            except DxlTimeout:
                pass  # Status Return Level 0/1 does not answer writes

    def sync_write(self, addr: int, n: int, items: list[tuple[int, bytes]]) -> None:
        params = bytearray(u16(addr) + u16(n))
        for dxl_id, data in items:
            if len(data) != n:
                raise ValueError("sync_write: wrong data length")
            params.append(dxl_id)
            params += data
        pkt = build_packet(BROADCAST, INST_SYNC_WRITE, bytes(params))
        self._send(pkt)

    def sync_read(self, addr: int, n: int, ids: list[int]) -> dict[int, bytes]:
        pkt = build_packet(BROADCAST, INST_SYNC_READ, u16(addr) + u16(n) + bytes(ids))
        self._send(pkt)
        deadline = self._deadline(len(pkt), len(ids) * (11 + n))
        out: dict[int, bytes] = {}
        while len(out) < len(ids):
            try:
                pid, _, data = self._status(deadline)
            except DxlTimeout:
                missing = [i for i in ids if i not in out]
                raise DxlTimeout(f"sync_read: no answer from {missing}")
            if pid in ids and len(data) == n:
                out[pid] = data
        return out

    def fast_sync_read(self, addr: int, n: int, ids: list[int]) -> dict[int, bytes]:
        pkt = build_packet(BROADCAST, INST_FAST_SYNC_READ, u16(addr) + u16(n) + bytes(ids))
        self._send(pkt)
        deadline = self._deadline(len(pkt), 10 + len(ids) * (n + 4))
        while True:
            pid, inst, p = self._next_packet(deadline)
            if inst == INST_STATUS and pid == BROADCAST:
                break
        # p = [err1,id1,data1,crc1, err2,id2,data2,crc2, ..., errN,idN,dataN]  (no byte stuffing)
        expected = len(ids) * (n + 4) - 2
        if len(p) != expected:
            raise DxlError(f"fast_sync_read: {len(p)} bytes, expected {expected}")
        out: dict[int, bytes] = {}
        i = 0
        for _ in ids:
            err, pid = p[i], p[i + 1]
            if err & 0x80:
                self.hw_alerts[pid] = err
            out[pid] = bytes(p[i + 2:i + 2 + n])
            i += n + 4
        return out

    def reboot(self, dxl_id: int) -> None:
        pkt = build_packet(dxl_id, INST_REBOOT)
        self._send(pkt)
        try:
            self._status(self._deadline(len(pkt), 11), dxl_id)
        except DxlTimeout:
            pass
