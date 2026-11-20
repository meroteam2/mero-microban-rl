"""Walking runtime state machine (hardware-agnostic; driven by app_run or tests)."""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from . import battery as bat
from .calibration import UPSTREAM_MOUNT
from .config import resolve
from .imu import quat_to_mat
from .joints import JOINT_RANGE, JOINTS, LABEL_KO, STAND_POSE
from .walklog import WalkLog

# Frame of the gyro the upstream walk.onnx was trained on: the MJCF `imu` site, i.e. the chip as
# Rhoban mounted it. Our chip can sit differently (calibrated mount_quat), so rotate the trunk-frame
# gyro into that frame instead of passing our raw chip axes.
_R_UPSTREAM_IMU = quat_to_mat(np.array(UPSTREAM_MOUNT, float))

IDLE, RAMP, STAND, WALK, STOPPING, FALLEN = "IDLE", "RAMP", "STAND", "WALK", "STOPPING", "FALLEN"
STATE_KO = {IDLE: "토크OFF", RAMP: "서는중", STAND: "서있음", WALK: "걷는중", STOPPING: "멈추는중", FALLEN: "넘어짐"}

# XL330 current proxy (upstream scheduler.py)
KT, RES, ERR_GAIN, I_MAX = 0.366, 2.811, 0.0028773775, 1.75


class WalkController:
    def __init__(self, cfg: dict, bus, imu, policy, monitor: bat.BatteryMonitor, emit=print,
                 clock=time.monotonic, vread=None):
        self.cfg = cfg
        self.c = cfg["control"]
        self.tp = cfg["teleop"]
        self.bus, self.imu, self.policy, self.mon = bus, imu, policy, monitor
        self._emit_raw = emit
        lc = cfg.get("log", {})
        self.log = WalkLog(resolve(lc.get("dir", "logs")), bool(lc.get("walk", True)))
        self.now = clock
        self.state = IDLE
        self.t_state = self.now()
        self.q: dict[str, float] = {}
        self.dq: dict[str, float] = {}
        self.cmd_des = np.zeros(3)     # vx, vy, wz requested by keys
        self.cmd = np.zeros(3)         # rate-limited command given to the policy
        self.last_targets: dict[str, float] = dict(STAND_POSE)
        self.ramp_from: dict[str, float] = {}
        self.ramp_to: dict[str, float] = dict(STAND_POSE)
        self.ramp_s = float(self.c["ramp_s"])
        self.stop_phase = 0
        self.confirm_until: dict[str, float] = {}
        self.read_errors = 0
        self.vread_last = -1e9
        self.vread_idx = 0
        self.vread = vread             # battery sensor (bat.voltage_reader); None = motor input voltage
        self.vread_errors = 0
        self.crit_deadline: float | None = None
        self.oc_ticks = 0
        self.cmd_hist: deque = deque(maxlen=4)
        self.ticks = 0
        self.quit = False
        self.lost = False
        self.walk_refused_reason = ""

    # ------------------------------------------------------------------ helpers
    def emit(self, msg: str) -> None:
        self.log.event(msg)
        self._emit_raw(msg)

    def _set(self, state: str) -> None:
        if state != self.state:
            old = self.state
            self.state = state
            self.t_state = self.now()
            if old in (WALK, STOPPING) and state not in (WALK, STOPPING):
                self.close_log(state)

    def _log_row(self, obs=None, act=None) -> None:
        if not self.log.active:
            return
        try:
            imu = self.imu.latest()
        except Exception:
            imu = None
        volt = self.mon.status.filtered_v if self.mon.status.samples else None
        self.log.row(self.state, self.cmd_des, self.cmd, imu, self.q, self.dq, obs, act,
                     self.last_targets, volt, self.read_errors)

    def close_log(self, why: str = "") -> None:
        if not self.log.active:
            return
        self._log_row()                      # last row carries the closing events (fall, torque off...)
        rows = self.log.rows
        p = self.log.close(why)
        self._emit_raw(f"걷기 로그 저장: {p} ({rows}줄)")

    def _confirm(self, key: str, msg: str, window: float = 2.0) -> bool:
        t = self.now()
        if self.confirm_until.get(key, 0) > t:
            self.confirm_until.pop(key, None)
            return True
        self.confirm_until[key] = t + window
        self.emit(msg)
        return False

    def torque_off(self, reason: str = "") -> None:
        self.bus.torque_off_all_robust()
        self.policy.reset()
        self.cmd[:] = 0
        self.cmd_des[:] = 0
        self.crit_deadline = None
        if reason:
            self.emit(reason)            # before the state change so the walk log still records it
        self._set(IDLE)

    def _start_ramp(self, to: dict[str, float], seconds: float) -> None:
        self.ramp_from = {n: self.q.get(n, to.get(n, 0.0)) for n in JOINTS}
        self.ramp_to = dict(to)
        self.ramp_s = seconds
        self._set(RAMP)

    def torque_on(self) -> None:
        try:
            self.q, self.dq, _ = self.bus.read_state()   # fresh pose: the ramp starts exactly here
        except Exception as e:
            self.emit(f"관절 값을 읽지 못했습니다: {e}")
            return
        if self.mon.status.level >= bat.CRITICAL:
            self.emit("배터리 위험 수준이라 토크를 켜지 않습니다. 충전하세요.")
            return
        out = [n for n in JOINTS
               if not (JOINT_RANGE[n][0] - 0.05 <= self.q[n] <= JOINT_RANGE[n][1] + 0.05)]
        if out:
            self.emit("관절 범위 밖이라 토크를 켜지 않습니다(켜는 순간 튈 수 있음): "
                      + ", ".join(f"{LABEL_KO[n]} {math.degrees(self.q[n]):+.0f}°" for n in out)
                      + "  → 자세를 바로잡거나 `python -m mero calib`로 원점/부호를 확인하세요")
            return
        bus = self.bus
        bus.no_profile()
        bus.set_kp(int(self.c["kp_stand"]))
        # hold where we are first so the ramp starts without a jump
        bus.write_goal({n: self.q[n] for n in JOINTS})
        bus.torque(True)
        self._start_ramp(STAND_POSE, float(self.c["ramp_s"]))
        self.emit("토크 ON → 서는 자세로 이동")

    def can_walk(self) -> str:
        if self.state != STAND:
            return "서 있는 상태(STAND)에서만 걷기를 시작할 수 있습니다"
        if self.mon.status.level >= bat.LOW:
            return f"배터리 {self.mon.status.filtered_v:.2f}V: 걷기 금지 (충전 필요)"
        s = self.imu.latest()
        if not s.valid or time.perf_counter() - s.t > 0.2:
            return "IMU 데이터가 없습니다"
        if s.proj_grav[2] > -0.85:
            return f"몸통이 {math.degrees(math.acos(min(1, -s.proj_grav[2]))):.0f}° 기울어져 있습니다"
        return ""

    def start_walk(self) -> None:
        why = self.can_walk()
        if why:
            self.emit(why)
            return
        self.policy.reset()
        self.cmd[:] = 0
        self.stop_phase = 0
        self.bus.set_kp(int(self.c["kp_walk"]))
        self.cmd_hist.clear()
        self.oc_ticks = 0
        self._set(WALK)
        pol = self.policy
        self.log.start(pol.order, pol.obs_size, {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": getattr(pol, "path", ""),
            "policy_meta": getattr(pol, "meta", {}), "order": pol.order, "obs_size": pol.obs_size,
            "act_size": pol.act_size, "action_scale": getattr(pol, "action_scale", None),
            "gyro_frame": pol.gyro_frame, "control": self.c, "teleop": self.tp,
            "calibration": {n: {"sign": c.sign, "zero_deg": round(math.degrees(c.zero), 3)}
                            for n, c in self.bus.cal.joints.items()},
            "imu_mount_quat": list(self.bus.cal.mount_quat), "gyro_bias": list(self.bus.cal.gyro_bias),
            "multiturn_joints": sorted(getattr(self.bus, "multiturn", [])),
        })
        self.emit("걷기 시작 (방향키로 속도 조절, SPACE로 정지)")

    def stop_walk(self, reason: str = "") -> None:
        if self.state != WALK:
            return
        self.cmd_des[:] = 0
        self.stop_phase = 0
        self._set(STOPPING)
        self.emit("걷기 정지 중..." + (f" ({reason})" if reason else ""))

    # ------------------------------------------------------------------ keys
    def handle_key(self, k: str) -> None:
        from . import term as T
        st = self.state
        if k in ("k", T.KEY_ESC, T.KEY_BACKSPACE):
            self.torque_off("비상 정지: 모든 토크 OFF")
        elif k == "t":
            if st in (IDLE, FALLEN):
                self.torque_on()
            elif st == RAMP:
                self.torque_off("토크 OFF")
            elif st == STAND:
                if self._confirm("t", "한 번 더 t를 누르면 토크 OFF (로봇이 주저앉습니다. 손으로 잡으세요)"):
                    self.torque_off("토크 OFF")
            else:
                self.emit("걷는 중에는 토크를 끌 수 없습니다. 먼저 SPACE로 멈추세요 (비상시 k)")
        elif k in (T.KEY_SPACE, "w"):
            if st == WALK:
                self.stop_walk()
            elif st == STAND:
                self.start_walk()
            elif st == STOPPING:
                self.emit("멈추는 중입니다")
            else:
                self.emit("먼저 t로 토크를 켜고 서게 하세요")
        elif k in (T.KEY_UP, T.KEY_DOWN, T.KEY_LEFT, T.KEY_RIGHT, "a", "d", "s", "x"):
            tp = self.tp
            if k == T.KEY_UP:
                self.cmd_des[0] = min(self.cmd_des[0] + tp["vx_step"], tp["vx_max"])
            elif k == T.KEY_DOWN:
                self.cmd_des[0] = max(self.cmd_des[0] - tp["vx_step"], tp["vx_min"])
            elif k == "a":
                self.cmd_des[1] = min(self.cmd_des[1] + tp["vy_step"], tp["vy_max"])
            elif k == "d":
                self.cmd_des[1] = max(self.cmd_des[1] - tp["vy_step"], -tp["vy_max"])
            elif k == T.KEY_LEFT:
                self.cmd_des[2] = min(self.cmd_des[2] + tp["wz_step"], tp["wz_max"])
            elif k == T.KEY_RIGHT:
                self.cmd_des[2] = max(self.cmd_des[2] - tp["wz_step"], -tp["wz_max"])
            else:
                self.cmd_des[:] = 0
            self.cmd_des = np.round(self.cmd_des, 3)
            if st != WALK:
                self.emit(f"속도 명령 vx={self.cmd_des[0]:+.2f} vy={self.cmd_des[1]:+.2f} wz={self.cmd_des[2]:+.2f} (걷기 시작하면 적용)")
        elif k == "q":
            if st in (IDLE, FALLEN) or self._confirm("q", "한 번 더 q를 누르면 토크를 끄고 종료합니다 (로봇이 주저앉습니다)"):
                self.quit = True
        elif k == T.KEY_EOF:
            self.connection_lost()

    def connection_lost(self) -> None:
        if self.lost:
            return
        self.lost = True
        self.emit("SSH 연결 끊김: 걷기를 멈추고 토크를 끕니다")
        if self.state == WALK:
            self.stop_walk("연결 끊김")
        elif self.state in (STAND, RAMP):
            self.quit = True

    # ------------------------------------------------------------------ battery
    def _battery(self, t: float) -> None:
        period = float(self.cfg["battery"]["read_period_s"])
        if t - self.vread_last < period:
            return
        self.vread_last = t
        try:
            if self.vread is not None:
                v = self.vread()
            else:
                name = JOINTS[self.vread_idx % len(JOINTS)]
                self.vread_idx += 1
                v = self.bus.read_voltage(name)
        except Exception as e:
            self.vread_errors += 1
            if self.vread is not None:
                self.mon.status.error = str(e)
                if self.vread_errors in (1, 5) or self.vread_errors % 60 == 0:
                    self.emit(f"\a⚠ 배터리 전압을 읽을 수 없습니다 ({e}): 배터리 보호 꺼짐")
            return
        if self.vread_errors >= 5:
            self.emit("배터리 측정 복구")
        self.vread_errors = 0
        s = self.mon.update(v, t)
        if s.changed:
            lvl = s.level
            if lvl == bat.WARN:
                self.emit(f"\a⚠ 배터리 주의 {s.filtered_v:.2f}V: 곧 충전하세요")
            elif lvl == bat.LOW:
                self.emit(f"\a⚠⚠ 배터리 부족 {s.filtered_v:.2f}V: 걷기를 멈추고 더 이상 시작하지 않습니다")
                self.stop_walk("배터리 부족")
            elif lvl == bat.CRITICAL:
                cd = float(self.cfg["battery"]["critical_countdown_s"])
                self.emit(f"\a\a⛔ 배터리 위험 {s.filtered_v:.2f}V: {cd:.0f}초 뒤 토크 OFF. 로봇을 잡고 전원을 끄세요")
                self.stop_walk("배터리 위험")
                if self.state != IDLE:
                    self.crit_deadline = t + cd
                try:
                    self.bus.led(True)
                except Exception:
                    pass
            elif lvl == bat.OK:
                self.emit(f"배터리 회복 {s.filtered_v:.2f}V")
                self.crit_deadline = None
        if self.crit_deadline is not None and self.state in (STAND, RAMP) and t >= self.crit_deadline:
            self.torque_off("⛔ 배터리 위험: 토크 OFF. 전원 스위치를 끄고 충전하세요")

    # ------------------------------------------------------------------ safety
    def _overcurrent(self, targets: dict[str, float]) -> bool:
        aligned = self.cmd_hist[0] if self.cmd_hist else targets
        vin = self.mon.status.filtered_v if self.mon.status.samples else 7.5
        kp = int(self.c["kp_walk"])
        total = 0.0
        for n, tgt in aligned.items():
            q = self.q.get(n)
            if q is None:
                continue
            duty = max(-1.0, min(1.0, kp * ERR_GAIN * (tgt - q)))
            total += min(abs((vin * duty - KT * self.dq.get(n, 0.0)) / RES), I_MAX)
        self.oc_ticks = self.oc_ticks + 1 if total >= float(self.c["overcurrent_a"]) else 0
        if self.oc_ticks >= 2:
            self.torque_off(f"과전류 추정 {total:.1f}A: 토크 OFF")
            return True
        return False

    # ------------------------------------------------------------------ main step
    def tick(self) -> None:
        t = self.now()
        self.ticks += 1
        st = self.state
        need_read = st != IDLE or self.ticks % 5 == 1 or not self.q
        if need_read:
            try:
                if st in (WALK, STOPPING) and self.q:
                    # head is not part of the policy: skip it to save ~1 ms of bus time
                    q, dq, _ = self.bus.read_state(self.policy.order)
                    self.q.update(q)
                    self.dq.update(dq)
                else:
                    self.q, self.dq, _ = self.bus.read_state()
                self.read_errors = 0
            except Exception as e:
                self.read_errors += 1
                if self.read_errors >= 3 and st != IDLE:
                    self.torque_off(f"모터 통신 오류 3회 연속 ({e}): 토크 OFF")
                return
        self._battery(t)
        st = self.state
        dt = 1.0 / float(self.c["hz"])

        if st == RAMP:
            a = min((t - self.t_state) / max(self.ramp_s, 1e-3), 1.0)
            a = 0.5 - 0.5 * math.cos(math.pi * a)   # smooth start/stop
            tgt = {n: self.ramp_from[n] + a * (self.ramp_to[n] - self.ramp_from[n]) for n in JOINTS}
            self.bus.write_goal(tgt)
            self.last_targets = tgt
            if t - self.t_state >= self.ramp_s:
                self._set(STAND)
                self.emit("서기 완료. SPACE: 걷기 시작")
        elif st == STAND:
            self.bus.write_goal(self.ramp_to)
            self.last_targets = dict(self.ramp_to)
        elif st in (WALK, STOPPING):
            stepped = False
            try:
                s = self.imu.latest()
                if (time.perf_counter() - s.t > 0.25 or not s.valid) and self.stop_phase == 0:
                    # no fresh IMU: the policy would act on stale tilt -> go straight to the stand pose
                    self.emit("IMU 데이터가 끊겼습니다: 바로 서는 자세로 전환")
                    self._set(STOPPING)
                    self.stop_phase = 1
                    self.ramp_from = {n: self.last_targets.get(n, STAND_POSE[n]) for n in self.policy.order}
                    self.bus.set_kp(int(self.c["kp_stand"]))
                    self.cmd_des[:] = 0
                if s.proj_grav[2] > float(self.c["fall_gravity_z"]):
                    if self.c.get("torque_off_on_fall", True):
                        self.torque_off("넘어짐 감지: 토크 OFF. 로봇을 세운 뒤 t")
                        self._set(FALLEN)
                    else:
                        self._set(FALLEN)
                        self.emit("넘어짐 감지: 정책 정지")
                    return
                # rate-limit the command (m/s^2, rad/s^2)
                lim = np.array([0.5, 0.5, 2.0]) * dt
                want = self.cmd_des if st == WALK else np.zeros(3)
                self.cmd += np.clip(want - self.cmd, -lim, lim)
                if st == STOPPING and self.stop_phase == 1:
                    a = min((t - self.t_state) / 0.6, 1.0)
                    tgt = {n: self.ramp_from[n] + a * (STAND_POSE[n] - self.ramp_from[n]) for n in self.ramp_from}
                    self.bus.write_goal(tgt)
                    self.last_targets = tgt
                    if a >= 1.0:
                        self.ramp_to = dict(STAND_POSE)
                        self._set(STAND)
                        self.emit("정지 완료 (서 있음)")
                        if self.lost:
                            self.quit = True
                    return
                gyro = _R_UPSTREAM_IMU.T @ s.gyro_body if self.policy.gyro_frame == "imu" else s.gyro_body
                tgt = self.policy.step(gyro, s.proj_grav, self.q, self.dq, self.cmd)
                stepped = True
                if st == WALK and self._overcurrent(tgt):
                    return
                self.cmd_hist.append(dict(tgt))
                self.bus.write_goal(tgt)
                self.last_targets = tgt
                if st == STOPPING and t - self.t_state >= float(self.c["stop_settle_s"]) and np.all(np.abs(self.cmd) < 1e-3):
                    self.stop_phase = 1
                    self.ramp_from = dict(tgt)
                    self.bus.set_kp(int(self.c["kp_stand"]))
                    self.t_state = t
            finally:
                pol = self.policy
                self._log_row(getattr(pol, "last_obs", None) if stepped else None,
                              pol.last_action if stepped else None)

    def status_text(self, hz: float) -> str:
        s = self.imu.latest()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(s.proj_grav[2]))))) if s.valid else float("nan")
        c = self.cmd if self.state in (WALK, STOPPING) else self.cmd_des
        return (f"[{STATE_KO[self.state]}] {hz:4.1f}Hz  vx{c[0]:+.2f} vy{c[1]:+.2f} wz{c[2]:+.2f}  "
                f"기울기{tilt:3.0f}°  {self.mon.line()}")
