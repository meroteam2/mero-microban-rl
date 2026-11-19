"""Make a Microban-Walk(-DR) checkpoint whose actor is microban's walk.onnx.

walk.onnx (MLP 63 -> 512 -> 256 -> 128 -> 18, ELU, observation normalizer) has
exactly our actor's architecture and observation / action layout, except that
its gyro is in the IMU frame while ours is in the trunk frame. The IMU mount
rotation is a signed axis permutation (gyro_imu[i] = s_i * gyro_trunk[p(i)]),
so it is folded into the normalizer and the first layer:

  mean'[p(i)] = s_i * mean[i],  std'[p(i)] = std[i],  W0'[:, p(i)] = s_i * W0[:, i]

The normalizer count is set huge so that training barely moves its statistics.
The critic, optimizer and everything else come from --template (any
checkpoint of the same task, e.g. its model_0.pt).

  uv run python scripts/init_from_walk_onnx.py --template <run>/model_0.pt --out <dir>/model_0.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from microban_rl.walk_onnx import IMU_MOUNT_QUAT, load_walk_mlp


@dataclass(frozen=True)
class Args:
  template: str
  out: str
  std: float = 0.15
  """Initial action noise std (rad)."""


def quat_to_mat(q) -> np.ndarray:
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])


def main(a: Args) -> None:
  mlp, _ = load_walk_mlp()
  ckpt = torch.load(a.template, map_location="cpu", weights_only=False)
  sd = ckpt["actor_state_dict"]

  # gyro_imu = R^T gyro_trunk (quat_apply_inverse of the mount quaternion).
  rt = quat_to_mat(IMU_MOUNT_QUAT).T
  assert np.allclose(np.abs(rt).sum(1), 1) and np.allclose(np.abs(rt), np.round(np.abs(rt))), rt
  perm = np.abs(rt).argmax(1)  # gyro_imu[i] = sign[i] * gyro_trunk[perm[i]]
  sign = rt[np.arange(3), perm]

  mean = mlp.mean.detach().clone().reshape(-1)
  std = mlp.std.detach().clone().reshape(-1)
  w0 = mlp.mlp[0].weight.detach().clone()
  new_mean, new_std, new_w0 = mean.clone(), std.clone(), w0.clone()
  for i in range(3):
    j = int(perm[i])
    new_mean[j] = float(sign[i]) * mean[i]
    new_std[j] = std[i]
    new_w0[:, j] = float(sign[i]) * w0[:, i]

  sd["obs_normalizer._mean"] = new_mean.reshape(1, -1)
  sd["obs_normalizer._std"] = new_std.reshape(1, -1)
  sd["obs_normalizer._var"] = new_std.square().reshape(1, -1)
  sd["obs_normalizer.count"] = torch.tensor(1e9, dtype=sd["obs_normalizer.count"].dtype)
  for k, layer in zip((0, 2, 4, 6), (0, 2, 4, 6)):
    lin = mlp.mlp[layer]
    sd[f"mlp.{k}.weight"] = (new_w0 if k == 0 else lin.weight.detach().clone())
    sd[f"mlp.{k}.bias"] = lin.bias.detach().clone()
  sd["distribution.std_param"] = torch.full_like(sd["distribution.std_param"], a.std)
  ckpt["iter"] = 0

  # Check: same output as walk.onnx on random observations (gyro rotated).
  x = torch.randn(256, 63)
  x_imu = x.clone()
  for i in range(3):
    x_imu[:, i] = float(sign[i]) * x[:, int(perm[i])]
  ref = mlp(x_imu)
  h = (x - sd["obs_normalizer._mean"]) / sd["obs_normalizer._std"]
  for k in (0, 2, 4):
    h = torch.nn.functional.elu(h @ sd[f"mlp.{k}.weight"].T + sd[f"mlp.{k}.bias"])
  out = h @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]
  print(f"gyro perm {perm.tolist()} sign {sign.tolist()}  max |ours - walk.onnx| = {(out - ref).abs().max():.2e}")

  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  torch.save(ckpt, a.out)
  print("saved", a.out)


if __name__ == "__main__":
  main(tyro.cli(Args))
