"""Non-blocking TCP server that streams robot state as JSON lines to viewers."""
from __future__ import annotations

import json
import socket
import time

from .joints import JOINTS


class StateStream:
    def __init__(self, port: int = 9870, hz: float = 20.0, host: str = "0.0.0.0"):
        self.period = 1.0 / hz
        self._last = 0.0
        self.clients: list[socket.socket] = []
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(4)
        self.srv.setblocking(False)
        self.port = port

    def _accept(self) -> list[str]:
        new = []
        while True:
            try:
                c, addr = self.srv.accept()
            except (BlockingIOError, OSError):
                break
            c.setblocking(False)
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.clients.append(c)
            new.append(f"{addr[0]}:{addr[1]}")
            self._send(c, {"type": "hello", "joints": JOINTS})
        return new

    def _send(self, c: socket.socket, msg: dict) -> bool:
        try:
            c.send((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            return True
        except (BlockingIOError, InterruptedError):
            return True   # viewer is slow; drop this frame
        except OSError:
            return False

    def due(self) -> bool:
        return time.monotonic() - self._last >= self.period

    def publish(self, msg: dict) -> list[str]:
        """Send one frame if the rate allows. Returns newly connected viewer addresses."""
        new = self._accept()
        if not self.due():
            return new
        self._last = time.monotonic()
        msg = dict(msg, type="state")
        self.clients = [c for c in self.clients if self._send(c, msg)]
        return new

    def close(self) -> None:
        for c in self.clients:
            try:
                c.close()
            except OSError:
                pass
        self.srv.close()


def state_msg(mode: str, q: dict, imu=None, ghost: dict | None = None, selected: str | None = None,
              volt: float | None = None, info: str = "") -> dict:
    m = {"mode": mode, "t": time.time(), "q": [round(float(q.get(n, 0.0)), 4) for n in JOINTS]}
    if imu is not None:
        m["quat_body"] = [round(float(v), 4) for v in imu.quat_body]
        m["proj_grav"] = [round(float(v), 4) for v in imu.proj_grav]
        m["gyro_body"] = [round(float(v), 4) for v in imu.gyro_body]
    if ghost is not None:
        m["ghost"] = [round(float(ghost.get(n, 0.0)), 4) for n in JOINTS]
    if selected:
        m["selected"] = selected
    if volt is not None:
        m["volt"] = round(volt, 2)
    if info:
        m["info"] = info
    return m
