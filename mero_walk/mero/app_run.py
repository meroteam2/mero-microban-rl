"""`python -m mero run`: walk the robot from an SSH keyboard."""
from __future__ import annotations

import signal
import sys
import time

from . import config, term
from .battery import BatteryMonitor, voltage_reader
from .bus import BusLock, open_bus
from .calibration import Calibration
from .controller import WALK, STOPPING, WalkController
from .imu import ImuFrames, open_imu
from .policy import WalkPolicy
from .stream import StateStream, state_msg

HELP = f"""{term.BOLD}MERO 걷기 조종{term.RST}
  t        토크 ON(서는 자세로 이동) / 서 있을 때 두 번 누르면 토크 OFF
  SPACE    걷기 시작 / 정지   (w도 같음)
  ↑ ↓      앞/뒤 속도         ← →  제자리/진행 중 회전
  a d      옆걸음 왼/오른      s    속도 0
  k, ESC   비상 정지(모든 토크 즉시 OFF)
  h        이 도움말           q    종료(두 번)
"""


def main(args) -> int:
    cfg = config.load(args.config)
    if args.hz:
        cfg["control"]["hz"] = args.hz
    if args.policy:
        cfg["policy"]["path"] = args.policy
    if args.gyro_frame:
        cfg["policy"]["gyro_frame"] = args.gyro_frame
    lock = BusLock(cfg["lock_file"])
    if not lock.acquire():
        print(f"다른 프로그램(PID {lock.owner()})이 모터 버스를 쓰고 있습니다. 먼저 종료하세요.")
        return 1
    cal = Calibration.load(config.resolve(cfg["calibration_file"]))
    if not cal.path.exists():
        print(f"{term.YEL}calibration.json이 없어 기본값(업스트림 부호, 원점 0)으로 실행합니다. "
              f"먼저 `python -m mero calib`로 확인하는 걸 권장합니다.{term.RST}")
    for line in cal.diff_from_default():
        print("  보정값:", line)
    print(f"모터 연결 중 ({cfg['serial']['port']} @ {cfg['serial']['baud']})...")
    try:
        bus = open_bus(cfg, cal)
        mode = bus.setup(cfg["serial"]["read_mode"])
    except Exception as e:
        print(f"{term.RED}모터 연결 실패: {e}{term.RST}\n  전원 스위치, 배선, config의 serial 설정, 권한(dialout)을 확인하세요.")
        lock.release()
        return 1
    print(f"  19개 응답, 읽기 방식 {mode}")
    frames = ImuFrames(cal.mount_quat, cal.gyro_bias)
    try:
        imu = open_imu(cfg, frames)
    except Exception as e:
        print(f"{term.RED}IMU 연결 실패: {e}{term.RST}\n  /dev/i2c-1 존재와 i2c 그룹 권한을 확인하세요.")
        bus.p.close()
        lock.release()
        return 1
    pol_path = config.resolve(cfg["policy"]["path"])
    policy = WalkPolicy(str(pol_path), cfg["policy"]["gyro_frame"], prefer_meta=not args.gyro_frame)
    src = "정책 메타데이터" if policy.gyro_from_meta else ("--gyro-frame" if args.gyro_frame else "config")
    print(f"  정책 {pol_path.name}: 관측 {policy.obs_size}, 행동 {policy.act_size}, gyro={policy.gyro_frame} ({src})")
    mon = BatteryMonitor(cfg)
    vread = voltage_reader(cfg)
    stream = None
    if not args.no_stream:
        try:
            stream = StateStream(int(cfg["stream"]["port"]), float(cfg["stream"]["hz"]))
        except OSError as e:
            print(f"  상태 스트림 포트 {cfg['stream']['port']} 사용 불가: {e}")
    status = term.StatusLine()
    ctl = WalkController(cfg, bus, imu, policy, mon, emit=status.event, vread=vread)
    signal.signal(signal.SIGHUP, lambda *_: ctl.connection_lost())
    signal.signal(signal.SIGTERM, lambda *_: setattr(ctl, "quit", True))

    hz = float(cfg["control"]["hz"])
    period = 1.0 / hz
    print(HELP)
    time.sleep(0.3)   # IMU filter settles
    rate_ema = hz
    overruns = 0
    last_overrun_msg = 0.0
    try:
        with term.Keyboard() as kb:
            t_next = time.perf_counter()
            last = t_next
            while not ctl.quit:
                for k in kb.keys(0.0):
                    if k in ("h", "?"):
                        status.event(HELP)
                    else:
                        ctl.handle_key(k)
                ctl.tick()
                now = time.perf_counter()
                rate_ema += 0.05 * (1.0 / max(now - last, 1e-4) - rate_ema)
                last = now
                if stream is not None:
                    for addr in stream.publish(state_msg("run", ctl.q, imu.latest(), volt=mon.status.filtered_v,
                                                         info=ctl.state)):
                        status.event(f"뷰어 연결됨: {addr}")
                status.show(ctl.status_text(rate_ema))
                t_next += period
                sl = t_next - time.perf_counter()
                if sl > 0:
                    time.sleep(sl)
                else:
                    t_next = time.perf_counter()
                    if ctl.state in (WALK, STOPPING):
                        overruns += 1
                        if overruns > 50 and now - last_overrun_msg > 10:
                            status.event(f"{term.YEL}제어 주기를 못 맞춥니다(실제 {rate_ema:.0f}Hz < {hz:.0f}Hz). "
                                         f"config의 control.hz를 낮추거나 `bench`를 보세요{term.RST}")
                            last_overrun_msg = now
    except KeyboardInterrupt:
        pass
    finally:
        status.clear()
        bus.torque_off_all_robust()
        ctl.close_log("exit")
        try:
            print("모든 토크 OFF. 종료합니다.")
        except OSError:
            pass
        imu.stop()
        if stream:
            stream.close()
        bus.p.close()
        lock.release()
    return 0
