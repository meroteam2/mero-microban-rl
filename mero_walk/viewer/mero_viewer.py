#!/usr/bin/env python3
"""MERO digital twin viewer (runs on the laptop).

Connects to `python -m mero calib` (or `run`) on the robot, which streams joint
angles and IMU tilt (calib: every bus read, ~30 Hz at 115200 bps), and shows the robot in Gazebo or MuJoCo:
  - solid model  = real robot (measured joints, trunk tilted by the IMU)
  - cyan ghost   = reference pose (calib 'z' mode) or jog target (calib 'j' mode)
  - orange ball  = selected joint (Gazebo)

  python3 viewer/mero_viewer.py --robot 192.168.137.23            # Gazebo (default)
  python3 viewer/mero_viewer.py --robot 192.168.137.23 --viewer mujoco
  python3 viewer/mero_viewer.py --demo                             # no robot: test the viewer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
SCENE = ROOT / "model" / "mjcf" / "scene.xml"

JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "head",
]
BASE_Z = 0.18


# ----------------------------------------------------------------------------- network
class StateClient(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.state: dict | None = None
        self.joints = list(JOINTS)
        self.connected = False
        self.frames = 0
        self.last_err = ""

    def run(self) -> None:
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=3) as s:
                    s.settimeout(3)
                    self.connected = True
                    buf = b""
                    while True:
                        chunk = s.recv(65536)
                        if not chunk:
                            raise ConnectionError("closed by robot")
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for ln in lines:
                            if not ln.strip():
                                continue
                            msg = json.loads(ln)
                            if msg.get("type") == "hello":
                                self.joints = msg["joints"]
                            elif msg.get("type") == "state":
                                with self.lock:
                                    self.state = msg
                                    self.frames += 1
            except (OSError, ValueError, ConnectionError) as e:
                self.connected = False
                self.last_err = str(e)
                time.sleep(1.0)

    def latest(self) -> dict | None:
        with self.lock:
            return self.state


class DemoSource:
    """Fake robot stream to check the viewer without hardware."""

    def __init__(self):
        self.t0 = time.time()
        self.connected = True
        self.frames = 0
        self.joints = list(JOINTS)
        self.last_err = ""

    def latest(self) -> dict:
        t = time.time() - self.t0
        self.frames += 1
        q = [0.0] * len(JOINTS)
        k = int(t / 3) % len(JOINTS)
        q[k] = 0.5 * math.sin(2 * math.pi * (t % 3) / 3)
        tilt = math.radians(15) * math.sin(0.5 * t)
        pg = [math.sin(tilt), 0.0, -math.cos(tilt)]
        return {"mode": "demo", "q": q, "proj_grav": pg, "selected": JOINTS[k],
                "ghost": [0.0] * len(JOINTS), "volt": 7.7}


# ----------------------------------------------------------------------------- kinematics
def quat_from_two_vectors(a, b) -> np.ndarray:
    a = np.asarray(a, float) / max(np.linalg.norm(a), 1e-9)
    b = np.asarray(b, float) / max(np.linalg.norm(b), 1e-9)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1.0 + d, *c])
    return q / np.linalg.norm(q)


class Kinematics:
    def __init__(self, scene: Path = SCENE):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d_live = mujoco.MjData(self.m)
        self.d_ghost = mujoco.MjData(self.m)
        self.qadr = {n: self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS}
        self.jid = {n: self.m.joint(n).id for n in JOINTS}
        self.bodies = [self.m.body(i).name for i in range(1, self.m.nbody)]

    def base_quat(self, msg: dict) -> np.ndarray:
        pg = msg.get("proj_grav")
        if not pg:
            return np.array([1.0, 0, 0, 0])
        # world <- body rotation with zero yaw that maps body gravity onto world -z
        return quat_from_two_vectors(pg, [0.0, 0.0, -1.0])

    def apply(self, d, q_list, names, base_q) -> None:
        d.qpos[:] = 0
        d.qpos[0:3] = [0.0, 0.0, BASE_Z]
        d.qpos[3:7] = base_q
        for n, v in zip(names, q_list):
            if n in self.qadr:
                d.qpos[self.qadr[n]] = v
        self.mj.mj_kinematics(self.m, d)

    def update(self, msg: dict, names: list[str]) -> bool:
        bq = self.base_quat(msg)
        self.apply(self.d_live, msg["q"], names, bq)
        ghost = msg.get("ghost")
        if ghost is not None:
            self.apply(self.d_ghost, ghost, names, bq)
        return ghost is not None

    def poses(self, d) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return [(self.m.body(i).name, d.xpos[i].copy(), d.xquat[i].copy()) for i in range(1, self.m.nbody)]

    def joint_anchor(self, name: str) -> np.ndarray:
        return self.d_live.xanchor[self.jid[name]].copy()


# ----------------------------------------------------------------------------- gazebo
class GazeboBackend:
    def __init__(self, kin: Kinematics, launch: bool = True, world_dir: Path | None = None, engine: str = "ogre2"):
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_v_pb2 import Pose_V
        import gz_world
        self.Pose_V, self.Boolean = Pose_V, Boolean
        self.kin = kin
        self.node = Node()
        self.world = (world_dir or Path.home() / ".cache" / "mero_viewer") / "mero_world.sdf"
        gz_world.build_world(self.world, engine=engine)
        self.svc = f"/world/{gz_world.WORLD_NAME}/set_pose_vector"
        self.proc = None
        if launch:
            gz = shutil.which("gz")
            if gz is None:
                raise SystemExit("gz 명령이 없습니다. viewer/README의 Gazebo Harmonic 설치를 먼저 하세요.")
            self.proc = subprocess.Popen([gz, "sim", "-r", "--render-engine", engine, str(self.world)])
            print(f"Gazebo 실행: gz sim -r {self.world}")
        self.ghost_visible = True
        self.marker_ok = True
        self._wait_service()
        try:
            from gz.msgs10.marker_pb2 import Marker
            from gz.msgs10.empty_pb2 import Empty
            self.Marker, self.Empty = Marker, Empty
        except ImportError:
            self.marker_ok = False

    def _wait_service(self, timeout: float = 60.0) -> None:
        t0 = time.time()
        print("Gazebo 월드를 기다리는 중...", end="", flush=True)
        while time.time() - t0 < timeout:
            if self.svc in self.node.service_list():
                ok, _ = self.node.request(self.svc, self.Pose_V(), self.Pose_V, self.Boolean, 2000)
                if ok:
                    print(" 연결됨")
                    return
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\nGazebo 서비스를 찾지 못했습니다. `gz sim`이 떠 있는지 확인하세요.")

    def _fill(self, req, prefix, poses, hidden=False) -> None:
        for name, pos, quat in poses:
            p = req.pose.add()
            p.name = prefix + name
            p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2] - (10 if hidden else 0))
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = (float(v) for v in quat)

    def update(self, msg: dict, has_ghost: bool) -> None:
        req = self.Pose_V()
        self._fill(req, "live_", self.kin.poses(self.kin.d_live))
        if has_ghost:
            self._fill(req, "ghost_", self.kin.poses(self.kin.d_ghost))
            self.ghost_visible = True
        elif self.ghost_visible:
            self._fill(req, "ghost_", self.kin.poses(self.kin.d_live), hidden=True)
            self.ghost_visible = False
        self.node.request(self.svc, req, self.Pose_V, self.Boolean, 50)
        sel = msg.get("selected")
        if sel and self.marker_ok:
            # the marker request blocks (up to 30 ms) - only send it when the ball actually moves
            pos = self.kin.joint_anchor(sel)
            now = time.monotonic()
            last = getattr(self, "_marker_last", None)
            if (last is None or sel != last[0]
                    or (np.linalg.norm(pos - last[1]) > 0.002 and now - last[2] > 0.1)):
                self._marker_last = (sel, pos, now)
                self._marker(pos)

    def _marker(self, pos) -> None:
        m = self.Marker()
        m.ns = "mero"
        m.id = 1
        m.action = self.Marker.ADD_MODIFY
        m.type = self.Marker.SPHERE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.022
        for c in (m.material.ambient, m.material.diffuse):
            c.r, c.g, c.b, c.a = 1.0, 0.45, 0.0, 1.0
        try:
            ok, _ = self.node.request("/marker", m, self.Marker, self.Empty, 30)
        except Exception:
            ok = False
        if not ok:
            self._marker_fail = getattr(self, "_marker_fail", 0) + 1
            if self._marker_fail > 20:
                self.marker_ok = False

    def alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# ----------------------------------------------------------------------------- mujoco
class MujocoBackend:
    def __init__(self, kin: Kinematics):
        import mujoco
        import mujoco.viewer
        self.mj = mujoco
        self.kin = kin
        self.v = mujoco.viewer.launch_passive(kin.m, kin.d_live, show_left_ui=False, show_right_ui=False)
        self.v.cam.lookat[:] = [0, 0, BASE_Z - 0.02]
        self.v.cam.distance = 0.75
        self.v.cam.azimuth = 45     # from behind-right: robot's left leg on the screen's left
        self.v.cam.elevation = -15
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[2] = 1   # visual meshes only
        self.pert = mujoco.MjvPerturb()

    def update(self, msg: dict, has_ghost: bool) -> None:
        mj = self.mj
        with self.v.lock():
            scn = self.v.user_scn
            scn.ngeom = 0
            if has_ghost:
                mj.mjv_addGeoms(self.kin.m, self.kin.d_ghost, self.opt, self.pert, mj.mjtCatBit.mjCAT_DYNAMIC, scn)
                for i in range(scn.ngeom):
                    scn.geoms[i].rgba[:] = [0.1, 0.8, 1.0, 0.3]
            sel = msg.get("selected")
            if sel and scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]),
                                self.kin.joint_anchor(sel), np.eye(3).flatten(), np.array([1.0, 0.45, 0.0, 1.0], np.float32))
                scn.ngeom += 1
        self.v.sync()

    def alive(self) -> bool:
        return self.v.is_running()

    def close(self) -> None:
        try:
            self.v.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=os.environ.get("MERO_ROBOT"),
                    help="로봇 IP 또는 호스트 이름 (예: 192.168.137.23, microban-01)")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--viewer", choices=["gazebo", "mujoco"], default="gazebo")
    ap.add_argument("--no-launch", action="store_true", help="이미 떠 있는 gz sim에 붙기")
    ap.add_argument("--demo", action="store_true", help="로봇 없이 가짜 움직임으로 뷰어만 확인")
    ap.add_argument("--hz", type=float, default=100.0, help="새 프레임 확인 주기(그리기는 새 프레임이 올 때만)")
    ap.add_argument("--render-engine", default="ogre2", choices=["ogre2", "ogre"],
                    help="WSL에서 화면이 안 나오면 ogre")
    a = ap.parse_args()
    if not a.demo and not a.robot:
        ap.error("--robot <로봇 IP>를 주세요 (또는 MERO_ROBOT 환경변수, 로봇 없이 확인은 --demo)")

    src = DemoSource() if a.demo else StateClient(a.robot, a.port)
    if not a.demo:
        src.start()
        print(f"로봇 {a.robot}:{a.port}에 연결 시도 (로봇에서 `python -m mero calib` 실행 중이어야 함)")
    kin = Kinematics()
    be = (GazeboBackend(kin, launch=not a.no_launch, engine=a.render_engine) if a.viewer == "gazebo"
          else MujocoBackend(kin))
    period = 1.0 / a.hz
    last_print = 0.0
    shown = -1
    try:
        while be.alive():
            t0 = time.time()
            msg = src.latest()
            # redraw only on a new robot frame: polling a ~30 Hz stream at a fixed 30 Hz showed
            # each frame for an uneven 1 or 2 ticks (judder)
            if msg and src.frames != shown:
                shown = src.frames
                has_ghost = kin.update(msg, src.joints)
                be.update(msg, has_ghost)
            if t0 - last_print > 1.0:
                last_print = t0
                if msg:
                    info = f"모드 {msg.get('mode')} {msg.get('info', '')}  선택 {msg.get('selected', '-')}  "
                    info += f"배터리 {msg.get('volt', float('nan')):.2f}V  프레임 {src.frames}"
                else:
                    info = f"대기 중... ({src.last_err or '연결 중'})"
                print("\r\033[K" + info, end="", flush=True)
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
