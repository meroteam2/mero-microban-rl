"""ONNX walking policy wrapper (mjlab_microban export format).

Observation (63 or 65 floats), as in upstream src/moves/walk.py:
  base_ang_vel(3) projected_gravity(3) joint_pos-default(18) joint_vel(18)
  last_action(18) command vx,vy,wz(3) [cos,sin reference phase(2)]
Joint order and default pose come from the ONNX metadata when present.
"""
from __future__ import annotations

import math

import numpy as np

from .joints import POLICY_ORDER_DEFAULT


class WalkPolicy:
    def __init__(self, path: str, gyro_frame: str = "imu"):
        self.path = str(path)
        self.last_obs = None
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.obs_size = int(self.sess.get_inputs()[0].shape[1])
        self.act_size = int(self.sess.get_outputs()[0].shape[1])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.meta = dict(meta)
        self.order = list(POLICY_ORDER_DEFAULT)
        default = {}
        if "joint_names" in meta and "default_joint_pos" in meta:
            names = meta["joint_names"].split(",")
            vals = [float(v) for v in meta["default_joint_pos"].split(",")]
            default = dict(zip(names, vals))
            ordered = [n for n in names if n != "head"]
            if len(ordered) == self.act_size:
                self.order = ordered
        if len(self.order) != self.act_size:
            raise ValueError(f"policy has {self.act_size} actions but {len(self.order)} joints are known")
        missing = [n for n in self.order if n not in default]
        if missing:
            raise ValueError(f"ONNX metadata has no default_joint_pos for {missing}")
        self.default = np.array([default[n] for n in self.order], dtype=np.float32)
        self.default_pose = {n: float(default[n]) for n in self.order}
        self.action_scale = float(meta.get("action_scale", "1.0").split(",")[0])
        base = 6 + 3 * self.act_size + 3
        if self.obs_size not in (base, base + 2):
            raise ValueError(f"unexpected observation size {self.obs_size} (expected {base} or {base + 2})")
        self.use_phase = self.obs_size == base + 2
        if gyro_frame not in ("imu", "body"):
            raise ValueError("gyro_frame must be 'imu' or 'body'")
        self.gyro_frame = gyro_frame
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(self.act_size, dtype=np.float32)
        self.phase_step = 0

    def build_obs(self, gyro: np.ndarray, proj_grav: np.ndarray, q: dict, dq: dict, cmd) -> np.ndarray:
        obs = np.empty(self.obs_size, dtype=np.float32)
        n = self.act_size
        obs[0:3] = gyro
        obs[3:6] = proj_grav
        obs[6:6 + n] = [q[j] for j in self.order]
        obs[6:6 + n] -= self.default
        obs[6 + n:6 + 2 * n] = [dq[j] for j in self.order]
        obs[6 + 2 * n:6 + 3 * n] = self.last_action
        obs[6 + 3 * n:9 + 3 * n] = cmd
        if self.use_phase:
            if np.mean(np.abs(cmd)) > 0.01:
                self.phase_step += 1
            else:
                self.phase_step = 0
            ph = (self.phase_step % 20) / 20 * 2 * math.pi
            obs[9 + 3 * n] = math.cos(ph)
            obs[10 + 3 * n] = math.sin(ph)
        return obs

    def step(self, gyro, proj_grav, q, dq, cmd) -> dict[str, float]:
        obs = self.build_obs(gyro, proj_grav, q, dq, cmd)
        act = self.sess.run(None, {self.input_name: obs[None, :]})[0][0].astype(np.float32)
        self.last_obs = obs               # kept for the walk log
        self.last_action = act
        tgt = self.default + act * self.action_scale
        return {j: float(tgt[k]) for k, j in enumerate(self.order)}
