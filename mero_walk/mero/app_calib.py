"""`python -m mero calib`: check/fix joint zero, motor direction and IMU mounting.

Run it on the robot over SSH, and open the viewer on the laptop
(viewer/mero_viewer.py) to see the digital twin move with the real robot.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from . import config, term
from .battery import BatteryMonitor
from .bus import BusLock, open_bus
from .calibration import IMU_POSES, Calibration, JointCal, solve_mount
from .imu import ImuFrames, open_imu
from .joints import DEFAULT_SIGN, JOINT_RANGE, JOINTS, LABEL_KO, MIRROR, MOTOR_ID, ZERO_POSE
from .stream import StateStream, state_msg

OBSERVE, JOG, ZERO, IMUWIZ = "관찰", "조그", "원점확인", "IMU"
JOG_STEP = math.radians(5.0)

HELP = (f"{term.DIM}↑↓ 관절 선택   f 부호 반전   j 조그 ON/OFF(선택 관절만 토크)   ←→ 조그 ±5°   "
        f"z 원점확인 모드   Z 선택 관절 원점=현재   A 전체 원점=현재\n"
        f"m IMU 장착방향 마법사   g 자이로 바이어스   r 선택 관절 기본값   s 저장   q 종료{term.RST}")


class CalibApp:
    def __init__(self, cfg, bus, imu, frames, cal: Calibration, stream=None, out=sys.stdout):
        self.cfg, self.bus, self.imu, self.frames, self.cal, self.stream = cfg, bus, imu, frames, cal, stream
        self.out = out
        self.sel = 0
        self.mode = OBSERVE
        self.jog_target: float | None = None
        self.q: dict[str, float] = {}
        self.raw = np.zeros(len(JOINTS))
        self.msg = "로봇을 손으로 움직여 보며 뷰어의 모델이 똑같이 움직이는지 확인하세요."
        self.dirty = False
        self.quit = False
        self.pending: str | None = None      # y/n confirmation key
        self.wiz_step = 0
        self.wiz_samples: dict[str, np.ndarray] = {}
        self.wiz_result: dict | None = None
        self.viewers = 0
        self.mon = BatteryMonitor(cfg)
        self._vlast = 0.0
        self._vidx = 0

    @property
    def joint(self) -> str:
        return JOINTS[self.sel]

    # ---------------------------------------------------------------- actions
    def _jog_off(self) -> None:
        if self.mode == JOG:
            self.bus.torque(False, [self.joint])
        self.jog_target = None

    def _jog_on(self) -> None:
        n = self.joint
        self.jog_target = self.q.get(n, 0.0)
        self.bus.set_kp(int(self.cfg["control"]["kp_stand"]), [n])
        self.bus.write_goal({n: self.jog_target})
        self.bus.torque(True, [n])

    def set_zero(self, names: list[str]) -> None:
        for n in names:
            c = self.cal.joints[n]
            motor_rad = (self.raw[JOINTS.index(n)] - 2048.0) * (2 * math.pi / 4096.0)
            # want sign*(motor_rad - zero) == ZERO_POSE[n]
            c.zero = motor_rad - c.sign * ZERO_POSE[n]
        self.dirty = True

    def handle_key(self, k: str) -> None:
        T = term
        if self.pending:
            what, self.pending = self.pending, None
            if k.lower() == "y":
                if what == "zero_all":
                    self.set_zero(JOINTS)
                    self.msg = "19개 관절 원점을 현재 자세(= 기준 자세)로 맞췄습니다. s로 저장하세요."
                elif what == "imu_apply":
                    r = self.wiz_result
                    self.cal.mount_quat = r["quat"]
                    self.frames.set_mount(r["quat"])
                    self.dirty = True
                    self.mode = OBSERVE
                    self.msg = f"IMU 장착 회전을 {tuple(round(v, 3) for v in r['quat'])}로 바꿨습니다. s로 저장하세요."
                elif what == "quit":
                    self.quit = True
            else:
                self.msg = "취소했습니다."
                if what == "imu_apply":
                    self.mode = OBSERVE
            return

        if self.mode == IMUWIZ:
            if k == T.KEY_ENTER and self.wiz_step < len(IMU_POSES):
                name = list(IMU_POSES)[self.wiz_step]
                self.msg = "측정 중... 움직이지 마세요"
                self.render(force=True)
                acc, _ = self.imu.average(1.5)
                self.wiz_samples[name] = acc
                self.wiz_step += 1
                if self.wiz_step < len(IMU_POSES):
                    nxt = list(IMU_POSES)[self.wiz_step]
                    self.msg = f"[{self.wiz_step + 1}/3] {IMU_POSES[nxt][1]} 후 ENTER"
                else:
                    self._finish_wizard()
            elif k in (T.KEY_ESC, "m"):
                self.mode = OBSERVE
                self.msg = "IMU 마법사 취소"
            return

        if k in (T.KEY_UP, T.KEY_DOWN):
            self._jog_off_keep_mode()
            self.sel = (self.sel + (-1 if k == T.KEY_UP else 1)) % len(JOINTS)
            if self.mode == JOG:
                self._jog_on()
            n = self.joint
            self.msg = f"선택: {LABEL_KO[n]} (ID {MOTOR_ID[n]}). 짝 관절: {LABEL_KO.get(MIRROR.get(n, ''), '-')}"
        elif k == "f":
            if self.mode == JOG:
                self.msg = "조그 중에는 부호를 바꿀 수 없습니다 (j로 조그를 끄세요)"
                return
            c = self.cal.joints[self.joint]
            c.sign = -c.sign
            self.dirty = True
            self.msg = f"{LABEL_KO[self.joint]} 부호 → {int(c.sign):+d} (기본 {int(DEFAULT_SIGN[self.joint]):+d})"
        elif k == "j":
            if self.mode == JOG:
                self._jog_off()
                self.mode = OBSERVE
                self.msg = "조그 OFF (토크 꺼짐)"
            else:
                self.mode = JOG
                self._jog_on()
                self.msg = (f"조그: {LABEL_KO[self.joint]}만 토크 ON. ←→로 5°씩 움직이면 뷰어 고스트(목표)와 "
                            f"실제 로봇이 같은 방향으로 가야 합니다")
        elif k in (T.KEY_LEFT, T.KEY_RIGHT) and self.mode == JOG:
            n = self.joint
            lo, hi = JOINT_RANGE[n]
            self.jog_target = min(max(self.jog_target + (JOG_STEP if k == T.KEY_RIGHT else -JOG_STEP), lo), hi)
            self.bus.write_goal({n: self.jog_target})
            self.msg = f"{LABEL_KO[n]} 목표 {math.degrees(self.jog_target):+.1f}° (+ 방향 = 시뮬레이션 양의 방향)"
        elif k == "z":
            self._jog_off()
            self.mode = OBSERVE if self.mode == ZERO else ZERO
            self.msg = ("원점확인: 로봇을 기준 자세(다리 쭉 펴고, 팔은 몸통 옆으로 곧게 내리고, 머리 정면)로 두세요. "
                        "'차이'가 기준과의 오차입니다." if self.mode == ZERO else "관찰 모드")
        elif k == "Z":
            self.set_zero([self.joint])
            self.msg = f"{LABEL_KO[self.joint]} 원점 = 현재 위치 ({math.degrees(self.cal.joints[self.joint].zero):+.1f}°)"
        elif k == "A":
            self.pending = "zero_all"
            self.msg = f"{term.YEL}지금 로봇이 정확히 기준 자세인가요? 19개 관절 원점을 모두 현재값으로 바꿉니다 [y/N]{term.RST}"
        elif k == "r":
            n = self.joint
            self.cal.joints[n] = JointCal(DEFAULT_SIGN[n], 0.0)
            self.dirty = True
            self.msg = f"{LABEL_KO[n]}을 기본값(부호 {int(DEFAULT_SIGN[n]):+d}, 원점 0)으로 되돌림"
        elif k == "m":
            self._jog_off()
            self.mode = IMUWIZ
            self.wiz_step = 0
            self.wiz_samples = {}
            first = list(IMU_POSES)[0]
            self.msg = f"IMU 마법사 [1/3] {IMU_POSES[first][1]} 후 ENTER (ESC 취소)"
        elif k == "g":
            self.msg = "자이로 바이어스 측정: 3초간 로봇을 가만히 두세요..."
            self.render(force=True)
            self.frames.gyro_bias = np.zeros(3)
            _, g = self.imu.average(3.0)
            self.frames.gyro_bias = g
            self.cal.gyro_bias = tuple(float(v) for v in g)
            self.dirty = True
            self.msg = f"자이로 바이어스 {np.degrees(g).round(2)} °/s 저장 대기 (s)"
        elif k == "s":
            p = self.cal.save()
            self.dirty = False
            self.msg = f"{term.GRN}저장함: {p}{term.RST}"
        elif k in ("q", T.KEY_EOF):
            if self.dirty and k == "q":
                self.pending = "quit"
                self.msg = f"{term.YEL}저장하지 않은 변경이 있습니다. 그래도 종료? [y/N]{term.RST}"
            else:
                self.quit = True
        elif k in ("k", T.KEY_ESC):
            self._jog_off()
            self.bus.torque_off_all_robust()
            self.mode = OBSERVE
            self.msg = "모든 토크 OFF"

    def _jog_off_keep_mode(self) -> None:
        if self.mode == JOG:
            self.bus.torque(False, [self.joint])

    def _finish_wizard(self) -> None:
        r = solve_mount(self.wiz_samples)
        self.wiz_result = r
        q_old = self.cal.mount_quat
        same = np.allclose(np.abs(np.dot(r["quat"], q_old)), 1.0, atol=1e-3)
        lines = [f"측정 오차 최대 {r['err_deg']:.1f}° " + ", ".join(f"{k} {v:.1f}°" for k, v in r["per_pose"].items())]
        if r["improper"]:
            lines.append(f"{term.RED}축 하나가 거울상으로 뒤집혀 있습니다(오른손 좌표계 아님). IMU 드라이버/칩 설정 문제입니다.{term.RST}")
        if not r["snapped"]:
            lines.append(f"{term.YEL}90° 단위 장착이 아닙니다(오차 큼). 자세를 더 정확히 하고 다시 해 보세요.{term.RST}")
        if same:
            lines.append(f"{term.GRN}현재 설정과 같습니다: IMU 장착 방향이 기대값과 맞습니다.{term.RST}")
            self.mode = OBSERVE
            self.msg = "\n".join(lines)
            return
        lines.append(f"새 장착 회전 {tuple(round(v, 3) for v in r['quat'])} (현재 {tuple(round(v, 3) for v in q_old)}). 적용? [y/N]")
        self.pending = "imu_apply"
        self.msg = "\n".join(lines)

    # ---------------------------------------------------------------- loop parts
    def update(self) -> None:
        try:
            self.q, _, self.raw = self.bus.read_state()
        except Exception as e:
            self.msg = f"{term.RED}모터 읽기 실패: {e}{term.RST}"
            return
        t = time.monotonic()
        if t - self._vlast > 2.0:
            self._vlast = t
            try:
                self.mon.update(self.bus.read_voltage(JOINTS[self._vidx % len(JOINTS)]))
                self._vidx += 1
            except Exception:
                pass
        if self.stream:
            ghost = None
            if self.mode == ZERO:
                ghost = ZERO_POSE
            elif self.mode == JOG and self.jog_target is not None:
                ghost = dict(self.q)
                ghost[self.joint] = self.jog_target
            new = self.stream.publish(state_msg("calib", self.q, self.imu.latest(), ghost=ghost,
                                                selected=self.joint, volt=self.mon.status.filtered_v))
            self.viewers = len(self.stream.clients)
            if new:
                self.msg = f"뷰어 연결됨: {', '.join(new)}"

    _last_render = 0.0

    def render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.2:
            return
        self._last_render = now
        L = []
        flag = f"{term.YEL}*저장 안 됨{term.RST}" if self.dirty else ""
        L.append(f"{term.BOLD}MERO 캘리브레이션{term.RST}  모드: {term.CYN}{self.mode}{term.RST}   "
                 f"뷰어 {self.viewers}대   {self.mon.line()}  {flag}")
        ref = self.mode == ZERO
        L.append(f"{'':2}{'ID':>3} {'관절':<15}{'raw':>6} {'각도°':>7} {'부호':>4} {'원점°':>6}"
                 + (f" {'차이°':>6}" if ref else f" {'범위°':>13}"))
        for i, n in enumerate(JOINTS):
            c = self.cal.joints[n]
            q = self.q.get(n, float("nan"))
            raw = int(self.raw[i]) if len(self.raw) else 0
            mark = ">" if i == self.sel else " "
            sign_col = term.YEL if c.sign != DEFAULT_SIGN[n] else ""
            row = (f"{mark} {MOTOR_ID[n]:>3} {LABEL_KO[n]:<15}{raw:>6} {math.degrees(q):>+7.1f} "
                   f"{sign_col}{int(c.sign):>+4d}{term.RST if sign_col else ''} {math.degrees(c.zero):>+6.1f}")
            lo, hi = JOINT_RANGE[n]
            if ref:
                d = math.degrees(q - ZERO_POSE[n])
                col = term.RED if abs(d) > 20 else (term.YEL if abs(d) > 5 else term.GRN)
                row += f" {col}{d:>+6.1f}{term.RST}"
            else:
                out = q < lo - 0.05 or q > hi + 0.05
                row += f" {term.RED if out else term.DIM}{math.degrees(lo):>+6.0f}~{math.degrees(hi):<+6.0f}{term.RST}"
                if out:
                    row += f" {term.RED}범위 밖{term.RST}"
            if self.mode == JOG and i == self.sel and self.jog_target is not None:
                row += f"  {term.CYN}목표 {math.degrees(self.jog_target):+.1f}°{term.RST}"
            if i == self.sel:
                row = term.BOLD + row + term.RST
            L.append(row)
        s = self.imu.latest()
        pg, gb = s.proj_grav, s.gyro_body
        roll = math.degrees(math.atan2(-pg[1], -pg[2]))
        pitch = math.degrees(math.asin(max(-1, min(1, pg[0]))))
        L.append(f"IMU 중력(몸통) {pg[0]:+.2f} {pg[1]:+.2f} {pg[2]:+.2f}  roll {roll:+5.1f}° pitch {pitch:+5.1f}°  "
                 f"gyro(몸통) x{gb[0]:+.2f} y{gb[1]:+.2f} z{gb[2]:+.2f}")
        L.append(f"{term.DIM}  기대: 서 있으면 중력≈(0,0,-1), 앞으로 숙이면 x+, 왼쪽으로 기울이면 y+, "
                 f"위에서 봐서 반시계로 돌리면 gyro z+{term.RST}")
        L.append("")
        L.append(self.msg)
        L.append("")
        L.append(HELP)
        try:
            self.out.write("\033[H\033[2J" + "\n".join(L) + "\n")
            self.out.flush()
        except OSError:
            self.quit = True


def main(args) -> int:
    cfg = config.load(args.config)
    lock = BusLock(cfg["lock_file"])
    if not lock.acquire():
        print(f"다른 프로그램(PID {lock.owner()})이 모터 버스를 쓰고 있습니다. 먼저 종료하세요.")
        return 1
    cal = Calibration.load(config.resolve(cfg["calibration_file"]))
    print("모터 연결 중...")
    try:
        bus = open_bus(cfg, cal)
        bus.setup(cfg["serial"]["read_mode"])
    except Exception as e:
        print(f"{term.RED}모터 연결 실패: {e}{term.RST}")
        lock.release()
        return 1
    bus.torque_off_all_robust()
    frames = ImuFrames(cal.mount_quat, cal.gyro_bias)
    try:
        imu = open_imu(cfg, frames)
    except Exception as e:
        print(f"{term.RED}IMU 연결 실패: {e}{term.RST}")
        bus.p.close()
        lock.release()
        return 1
    stream = None
    try:
        stream = StateStream(int(cfg["stream"]["port"]), float(cfg["stream"]["hz"]))
    except OSError as e:
        print("상태 스트림을 열지 못했습니다:", e)
    app = CalibApp(cfg, bus, imu, frames, cal, stream)
    try:
        with term.Keyboard() as kb:
            while not app.quit:
                for k in kb.keys(0.0):
                    app.handle_key(k)
                app.update()
                app.render()
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        bus.torque_off_all_robust()
        imu.stop()
        if stream:
            stream.close()
        bus.p.close()
        lock.release()
        print("\n모든 토크 OFF. 종료.")
    return 0
