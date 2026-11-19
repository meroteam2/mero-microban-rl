"""MERO Microban robot tools.

  python -m mero check        모터 설정 점검 (--fix: 권장값 기록)
  python -m mero bench        115200 버스에서 낼 수 있는 제어 주기 측정
  python -m mero battery      배터리 전압 (--watch: 백그라운드 감시)
  python -m mero imu          IMU 값 보기
  python -m mero calib        원점/회전방향/IMU 방향 확인·수정 (+ 노트북 뷰어 스트림)
  python -m mero run          SSH 키보드로 걷기 조종
  python -m mero set-baud N   모든 모터 통신 속도 변경
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mero", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config.json 경로 (기본: 패키지 옆)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--fix", action="store_true")
    b = sub.add_parser("bench"); b.add_argument("-n", type=int, default=200)
    bt = sub.add_parser("battery"); bt.add_argument("--watch", action="store_true")
    bt.add_argument("--period", type=float, default=20.0)
    bt.add_argument("--poweroff", action="store_true", help="위험 전압 60초 지속 시 오렌지파이 종료(root 필요)")
    sub.add_parser("imu")
    sub.add_parser("calib")
    r = sub.add_parser("run")
    r.add_argument("--hz", type=float, default=None)
    r.add_argument("--policy", default=None, help="ONNX 경로 (기본 models/walk.onnx)")
    r.add_argument("--gyro-frame", choices=["imu", "body"], default=None)
    r.add_argument("--no-stream", action="store_true")
    sb = sub.add_parser("set-baud"); sb.add_argument("new_baud", type=int)
    sub.add_parser("swap-leg-ids", help="왼/오른 다리 모터 ID(11~16 <-> 21~26) 맞바꾸기")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .app_run import main as m
    elif a.cmd == "calib":
        from .app_calib import main as m
    else:
        from . import app_tools as t
        m = {"check": t.cmd_check, "bench": t.cmd_bench, "battery": t.cmd_battery, "imu": t.cmd_imu,
             "set-baud": t.cmd_set_baud, "swap-leg-ids": t.cmd_swap_leg_ids}[a.cmd]
    return m(a) or 0


if __name__ == "__main__":
    sys.exit(main())
