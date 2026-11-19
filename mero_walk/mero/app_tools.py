"""Small robot-side tools: check, bench, battery, imu, set-baud."""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import time

import numpy as np

from . import config, dxl, term
from .battery import LEVEL_KO, BatteryMonitor, CRITICAL, LOW, WARN, OK, soc_percent
from .bus import BusLock, MotorBus, open_bus
from .calibration import Calibration
from .imu import ImuFrames, open_imu
from .joints import ID_TO_JOINT, JOINTS, LABEL_KO, MOTOR_ID, POLICY_ORDER_DEFAULT


def _locked(cfg) -> BusLock | None:
    lock = BusLock(cfg["lock_file"])
    if not lock.acquire():
        print(f"다른 프로그램(PID {lock.owner()})이 모터 버스를 쓰고 있습니다.")
        return None
    return lock


# ---------------------------------------------------------------------------- check
def cmd_check(args) -> int:
    """Read every motor's settings and flag anything that differs from what the runtime expects."""
    cfg = config.load(args.config)
    lock = _locked(cfg)
    if not lock:
        return 1
    cal = Calibration.load(config.resolve(cfg["calibration_file"]))
    bus = open_bus(cfg, cal)
    p = bus.p
    problems = 0
    print(f"{cfg['serial']['port']} @ {cfg['serial']['baud']} 에서 19개 모터 확인\n")
    print(f"{'ID':>3} {'관절':<16} {'모델':<11} {'FW':>3} {'모드':>4} {'Drive':>5} {'Homing':>6} "
          f"{'RDT':>3} {'Shut':>4} {'Lim(min-max)':>12} {'V':>4} {'°C':>3} {'HWerr':>5} 위치(°)")
    try:
        for n in JOINTS:
            mid = MOTOR_ID[n]
            r = p.ping(mid) or p.ping(mid)
            if r is None:
                print(f"{mid:>3} {LABEL_KO[n]:<16} {term.RED}응답 없음{term.RST}")
                problems += 1
                continue
            model, fw = r
            ee = p.read(mid, 0, 64)
            ram = p.read(mid, 64, 84)   # 64..147
            u8 = lambda a: (ee if a < 64 else ram)[a if a < 64 else a - 64]
            u16 = lambda a: struct.unpack_from("<H", ee if a < 64 else ram, a if a < 64 else a - 64)[0]
            s32 = lambda a: struct.unpack_from("<i", ee if a < 64 else ram, a if a < 64 else a - 64)[0]
            opm, drive, homing = u8(11), u8(10), s32(20)
            rdt, shut = u8(9), u8(63)
            pmax, pmin = s32(48), s32(52)
            volt, temp, hwerr = u16(144) * 0.1, u8(146), u8(70)
            pos = s32(132)
            q = cal.joints[n].to_sim(pos)
            flags = []
            if model not in (1200,):
                flags.append(f"모델 {dxl.MODEL_NAMES.get(model, model)} (XL330-M288 아님)")
            if opm != 3:
                flags.append(f"Operating Mode {opm} (3이어야 함)")
            if drive & 0x01:
                flags.append("Drive Mode 역방향 비트 ON (소프트웨어 부호와 겹침)")
            if homing != 0:
                flags.append(f"Homing Offset {homing} (0 권장, 원점은 calibration.json에서)")
            if rdt != 0:
                flags.append(f"Return Delay {rdt} (0 권장)")
            if shut != 52:
                flags.append(f"Shutdown {shut} (52 권장: 입력전압 오류 끔)")
            if (pmin, pmax) != (0, 4095):
                flags.append(f"위치 제한 {pmin}~{pmax} (0~4095 아니면 목표가 잘림)")
            if fw < 46:
                flags.append(f"펌웨어 v{fw} < 46 (Fast Sync Read 불가, 느려짐)")
            if hwerr:
                flags.append(f"하드웨어 에러 0x{hwerr:02x} (재부팅 필요)")
            col = term.RED if flags else ""
            print(f"{col}{mid:>3} {LABEL_KO[n]:<16} {dxl.MODEL_NAMES.get(model, str(model)):<11} {fw:>3} {opm:>4} "
                  f"{drive:>5} {homing:>6} {rdt:>3} {shut:>4} {pmin:>5}-{pmax:<6} {volt:>4.1f} {temp:>3} "
                  f"{hwerr:>5} {math.degrees(q):+7.1f}{term.RST if flags else ''}")
            for f in flags:
                print(f"      {term.YEL}↳ {f}{term.RST}")
            problems += len(flags)
            if args.fix and flags:
                _fix(p, mid, opm, drive, homing, rdt, shut, pmin, pmax)
        print()
        others = {i: v for i, v in p.broadcast_ping(0.2).items() if i not in ID_TO_JOINT}
        if others:
            print(f"{term.YEL}목록에 없는 ID도 응답: {sorted(others)}{term.RST}")
            problems += 1
    finally:
        p.close()
        lock.release()
    if problems:
        print(f"{term.YEL}확인할 항목 {problems}개.{term.RST}" + ("" if args.fix else " `check --fix`로 Operating Mode, Drive 역방향, "
              "Homing Offset, Return Delay, Shutdown, 위치 제한을 권장값으로 고칠 수 있습니다(토크 OFF 상태에서 EEPROM 기록)."))
    else:
        print(f"{term.GRN}모든 모터 설정이 런타임 기대값과 같습니다.{term.RST}")
    return 0


def _fix(p, mid, opm, drive, homing, rdt, shut, pmin, pmax) -> None:
    p.write(mid, dxl.ADDR_TORQUE_ENABLE, b"\x00")
    time.sleep(0.01)
    if opm != 3:
        p.write(mid, dxl.ADDR_OPERATING_MODE, b"\x03")
    if drive & 0x01:
        p.write(mid, dxl.ADDR_DRIVE_MODE, bytes([drive & ~0x01]))
    if homing != 0:
        p.write(mid, dxl.ADDR_HOMING_OFFSET, dxl.s32(0))
    if rdt != 0:
        p.write(mid, dxl.ADDR_RETURN_DELAY, b"\x00")
    if shut != 52:
        p.write(mid, dxl.ADDR_SHUTDOWN, bytes([52]))
    if (pmin, pmax) != (0, 4095):
        p.write(mid, dxl.ADDR_MIN_POSITION, dxl.s32(0))
        p.write(mid, dxl.ADDR_MAX_POSITION, dxl.s32(4095))
    print(f"      {term.GRN}↳ ID {mid} 권장값으로 기록함{term.RST}")


# ---------------------------------------------------------------------------- bench
def cmd_bench(args) -> int:
    """Measure the real read+write loop rate for each bus method (torque stays OFF)."""
    cfg = config.load(args.config)
    lock = _locked(cfg)
    if not lock:
        return 1
    cal = Calibration.load(config.resolve(cfg["calibration_file"]))
    bus = open_bus(cfg, cal)
    try:
        bus.setup("auto")
        best = bus.read_mode
        print(f"{cfg['serial']['port']} @ {cfg['serial']['baud']} bps, 토크 OFF 상태로 측정 (goal = 현재 위치)\n")
        modes = ["sync", "fast", "fast_indirect"]
        results: dict[str, float] = {}
        for mode in modes:
            if mode == "fast_indirect" and best != "fast_indirect":
                print(f"  {mode:<14} 사용 불가 (펌웨어/설정)")
                continue
            if mode == "fast" and best == "sync":
                print(f"  {mode:<14} 사용 불가 (펌웨어 < 46)")
                continue
            bus.read_mode = mode
            joints = POLICY_ORDER_DEFAULT
            times, errors = [], 0
            for _ in range(args.n):
                t0 = time.perf_counter()
                try:
                    q, _, _ = bus.read_state(joints)
                    bus.write_goal(q)
                except dxl.DxlError:
                    errors += 1
                    continue
                times.append(time.perf_counter() - t0)
            if not times:
                print(f"  {mode:<14} 실패 (오류 {errors}회)")
                continue
            t = np.array(times) * 1000
            results[mode] = float(np.percentile(t, 95))
            print(f"  {mode:<14} 평균 {t.mean():5.1f} ms, p95 {np.percentile(t, 95):5.1f} ms, 최대 {t.max():5.1f} ms "
                  f"→ 최대 약 {1000 / np.percentile(t, 95):4.1f} Hz  (오류 {errors}/{args.n})")
        bus.read_mode = best
        if best in results:
            # leave ~10% for the policy, IMU and the occasional voltage read
            hz = int(0.9 * 1000.0 / results[best])
            print(f"\n런타임은 '{best}'를 씁니다. 권장 control.hz ≈ {hz} "
                  f"(config.json에 {{\"control\": {{\"hz\": {hz}}}}}).")
            if hz < 50:
                print("  walk.onnx는 50 Hz로 학습됐습니다. 시뮬레이션 시험에선 25~40 Hz도 걸었지만, 실물에서 불안하면 "
                      "USB 어댑터(U2D2/OpenRB-150) + 1 Mbps(`set-baud 1000000`)로 50 Hz를 쓰거나, 이 주기로 정책을 다시 학습하세요.")
    finally:
        bus.p.close()
        lock.release()
    return 0


# ---------------------------------------------------------------------------- battery
def cmd_battery(args) -> int:
    cfg = config.load(args.config)
    if args.watch:
        return _battery_watch(cfg, args)
    lock = _locked(cfg)
    if not lock:
        st = _read_state_file(cfg)
        if st:
            print(f"런타임이 측정 중: {st['filtered_v']}V ({st['level']})")
        return 1
    bus = open_bus(cfg, Calibration())
    try:
        vs = {}
        for n in JOINTS:
            try:
                vs[n] = bus.read_voltage(n) + float(cfg["battery"]["offset_v"])
            except dxl.DxlError:
                pass
        if not vs:
            print("응답하는 모터가 없습니다")
            return 1
        mon = BatteryMonitor(cfg)
        med = float(np.median(list(vs.values())))
        lvl = mon._level_for(med, OK)
        pct = soc_percent(med, int(cfg["battery"]["cells"]))
        col = {OK: term.GRN, WARN: term.YEL, LOW: term.YEL, CRITICAL: term.RED}[lvl]
        for n, v in vs.items():
            print(f"  {MOTOR_ID[n]:>2} {LABEL_KO[n]:<16} {v:5.2f} V")
        b = cfg["battery"]
        print(f"\n{col}배터리(모터 중앙값) {med:.2f} V  ≈{pct:.0f}%  [{LEVEL_KO[lvl]}]{term.RST}  "
              f"(주의 {b['warn_v']} / 부족 {b['low_v']} / 위험 {b['critical_v']} V, {b['cells']}S)")
    finally:
        bus.p.close()
        lock.release()
    return 0


def _read_state_file(cfg) -> dict | None:
    try:
        with open(cfg["battery"]["state_file"]) as f:
            d = json.load(f)
        if time.time() - d["time"] < 10:
            return d
    except (OSError, ValueError, KeyError):
        pass
    return None


def _notify(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    try:
        subprocess.run(["wall", msg], timeout=3, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _battery_watch(cfg, args) -> int:
    """Background guard while nothing else runs: wall messages + red LEDs, optional poweroff."""
    b = cfg["battery"]
    mon = BatteryMonitor(cfg)
    mon.state_file = None
    mon.offset = 0.0   # offset is applied when reading the motors below
    last_level, last_msg, crit_since = OK, 0.0, None
    period = args.period
    print(f"배터리 감시 시작 ({period:.0f}초 간격, 주의 {b['warn_v']} / 부족 {b['low_v']} / 위험 {b['critical_v']} V)", flush=True)
    while True:
        v = None
        st = _read_state_file(cfg)
        if st:  # run/calib owns the bus and publishes the voltage
            v = float(st["filtered_v"])
        else:
            lock = BusLock(cfg["lock_file"])
            if lock.acquire():
                try:
                    bus = open_bus(cfg, Calibration())
                    vals = []
                    for n in ("left_ankle_roll", "right_ankle_roll", "head"):
                        try:
                            vals.append(bus.read_voltage(n))
                        except dxl.DxlError:
                            pass
                    if vals:
                        v = float(np.median(vals)) + float(b["offset_v"])
                        lvl_now = mon._level_for(v, mon.status.level)
                        try:
                            bus.led(lvl_now >= LOW)
                        except dxl.DxlError:
                            pass
                    bus.p.close()
                except Exception as e:  # serial port missing etc.
                    print("읽기 실패:", e, flush=True)
                finally:
                    lock.release()
        if v is not None:
            s = mon.update(v)
            now = time.monotonic()
            cycles = getattr(mon, "_cycles", 0) + 1
            mon._cycles = cycles
            if cycles % max(1, int(300 / period)) == 1:
                print(time.strftime("%H:%M:%S"), f"{s.filtered_v:.2f}V {LEVEL_KO[s.level]}"
                      + (" (런타임 측정값)" if st else ""), flush=True)
            if s.level != last_level or (s.level >= WARN and now - last_msg > 300):
                if s.level >= WARN:
                    _notify(f"[MERO 배터리] {s.filtered_v:.2f}V {LEVEL_KO[s.level]} — 충전하거나 전원을 끄세요")
                elif last_level >= WARN:
                    _notify(f"[MERO 배터리] {s.filtered_v:.2f}V 회복")
                last_level, last_msg = s.level, now
            if s.level >= CRITICAL:
                crit_since = crit_since or now
                if args.poweroff and now - crit_since > 60 and not st:
                    _notify("[MERO 배터리] 위험 전압 60초 지속: 오렌지파이를 종료합니다. 모터 전원은 스위치로 끄세요!")
                    subprocess.run(["systemctl", "poweroff"], check=False)
            else:
                crit_since = None
        time.sleep(period)


# ---------------------------------------------------------------------------- imu
def cmd_imu(args) -> int:
    cfg = config.load(args.config)
    cal = Calibration.load(config.resolve(cfg["calibration_file"]))
    imu = open_imu(cfg, ImuFrames(cal.mount_quat, cal.gyro_bias))
    try:
        time.sleep(0.5)
        while True:
            s = imu.latest()
            pg, gb = s.proj_grav, s.gyro_body
            roll = math.degrees(math.atan2(-pg[1], -pg[2]))
            pitch = math.degrees(math.asin(max(-1, min(1, pg[0]))))
            print(f"acc(칩) {s.acc_imu[0]:+.2f} {s.acc_imu[1]:+.2f} {s.acc_imu[2]:+.2f} g | "
                  f"중력(몸통) {pg[0]:+.2f} {pg[1]:+.2f} {pg[2]:+.2f} | roll {roll:+5.1f}° pitch {pitch:+5.1f}° | "
                  f"gyro(몸통) {gb[0]:+.2f} {gb[1]:+.2f} {gb[2]:+.2f} rad/s | 오류 {s.errors}", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()
    return 0


# ---------------------------------------------------------------------------- set-baud
def cmd_set_baud(args) -> int:
    """Change the baud rate of all 19 motors (EEPROM), e.g. to 1 Mbps for a USB adapter."""
    codes = {v: k for k, v in dxl.BAUD_CODES.items()}
    if args.new_baud not in codes:
        print("가능한 값:", sorted(codes))
        return 1
    cfg = config.load(args.config)
    lock = _locked(cfg)
    if not lock:
        return 1
    bus = open_bus(cfg, Calibration())
    try:
        found = bus.scan()
        print(f"현재 {cfg['serial']['baud']} bps에서 {len(found)}개 응답.")
        ans = input(f"모든 모터를 {args.new_baud} bps로 바꿉니다. 이후엔 config.json의 serial.baud도 바꿔야 합니다. 진행? [y/N] ")
        if ans.strip().lower() != "y":
            return 1
        bus.torque_off_all_robust()
        bus.p.sync_write(dxl.ADDR_BAUD, 1, [(i, bytes([codes[args.new_baud]])) for i in found])
        print("기록함. 전원을 껐다 켤 필요는 없지만, 새 속도로 `check`를 실행해 확인하세요.")
    finally:
        bus.p.close()
        lock.release()
    return 0
