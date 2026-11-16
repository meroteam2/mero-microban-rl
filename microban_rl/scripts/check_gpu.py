"""Sanity check: PyTorch and Warp both run kernels on the GPU."""

import torch
import warp as wp

print("torch", torch.__version__, "| CUDA", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch arch list", torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda")
print("torch matmul ok:", bool(torch.isfinite(x @ x).all()))


@wp.kernel
def double_index(a: wp.array(dtype=float)):
  i = wp.tid()
  a[i] = float(i) * 2.0


wp.init()
a = wp.zeros(8, dtype=float, device="cuda:0")
wp.launch(double_index, dim=8, inputs=[a])
print("warp", wp.__version__, "| kernel ok:", a.numpy().tolist())
