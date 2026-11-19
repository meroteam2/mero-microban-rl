"""Non-blocking keyboard input over SSH and simple terminal output helpers."""
from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT = "UP", "DOWN", "RIGHT", "LEFT"
KEY_ESC, KEY_ENTER, KEY_SPACE, KEY_BACKSPACE, KEY_EOF = "ESC", "ENTER", "SPACE", "BACKSPACE", "EOF"
_SEQ = {"[A": KEY_UP, "[B": KEY_DOWN, "[C": KEY_RIGHT, "[D": KEY_LEFT,
        "OA": KEY_UP, "OB": KEY_DOWN, "OC": KEY_RIGHT, "OD": KEY_LEFT}

BELL = "\a"
RED, YEL, GRN, CYN, DIM, BOLD, RST = "\033[31m", "\033[33m", "\033[32m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"


class Keyboard:
    """cbreak-mode stdin reader: Ctrl-C still raises KeyboardInterrupt."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.is_tty = os.isatty(self.fd)
        self._old = None
        self._pending = ""

    def __enter__(self) -> "Keyboard":
        if self.is_tty:
            self._old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            except termios.error:
                pass
            self._old = None

    def _read_available(self, timeout: float) -> str | None:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return ""
        data = os.read(self.fd, 64)
        if not data:
            return None   # EOF: the SSH session went away
        return data.decode("utf-8", errors="ignore")

    def keys(self, timeout: float = 0.0) -> list[str]:
        """All keys typed since the last call (non-blocking by default)."""
        data = self._read_available(timeout)
        if data is None:
            return [KEY_EOF]
        buf = self._pending + data
        self._pending = ""
        out = []
        i = 0
        while i < len(buf):
            c = buf[i]
            if c == "\x1b":
                if i + 1 >= len(buf):
                    # lone ESC, or the rest of an arrow sequence is still in flight
                    more = self._read_available(0.03)
                    if more:
                        buf += more
                        continue
                    out.append(KEY_ESC)
                    i += 1
                    continue
                seq = buf[i + 1:i + 3]
                if seq in _SEQ:
                    out.append(_SEQ[seq])
                    i += 3
                    continue
                out.append(KEY_ESC)
                i += 1
                continue
            if c in ("\r", "\n"):
                out.append(KEY_ENTER)
            elif c == " ":
                out.append(KEY_SPACE)
            elif c in ("\x7f", "\x08"):
                out.append(KEY_BACKSPACE)
            else:
                out.append(c)
            i += 1
        return out


class StatusLine:
    """One self-overwriting status line below scrolling event messages."""

    def __init__(self, stream=sys.stdout):
        self.s = stream
        self._last = 0.0
        self._text = ""
        self.dead = False   # terminal gone (SSH dropped): keep running silently

    def _w(self, text: str) -> None:
        if self.dead:
            return
        try:
            self.s.write(text)
            self.s.flush()
        except (OSError, ValueError):
            self.dead = True

    def event(self, msg: str) -> None:
        self._w("\r\033[K" + msg + "\n" + (self._text or ""))

    def show(self, text: str, min_period: float = 0.2, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < min_period:
            return
        self._last = now
        self._text = text
        self._w("\r\033[K" + text)

    def clear(self) -> None:
        self._w("\r\033[K")
        self._text = ""
