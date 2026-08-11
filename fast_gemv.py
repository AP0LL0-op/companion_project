"""Work around rocBLAS's bad kernel selection for batch-1 GEMV on gfx1030.

During autoregressive decode every nn.Linear sees a single token, so each
matmul is [1, K] @ [K, N] - a GEMV. rocBLAS picks a MT128x256x16 macro-tile
GEMM kernel for these. At M=1, N=1024 that yields a grid of just 4 workgroups
on a 72-CU GPU (~5% utilization), each doing a long serial K reduction. The
MLP down_proj ([1,8192]@[8192,1024]) lands worst: 629us against a ~39us
bandwidth roofline, and it alone was 61% of all matmul time.

torch.einsum('ij,j->i', W, x) sidesteps the heuristic and is faster on every
decode shape measured on this card:

    down_proj  [1,8192]@[8192,1024]   584.6us -> 32.5us   18.0x
    up/gate    [1,1024]@[1024,8192]   142.8us -> 51.6us    2.8x
    q/o_proj   [1,1024]@[1024,1024]    14.9us ->  9.1us    1.7x
    kv_proj    [1,1024]@[1024, 256]    13.7us ->  5.5us    2.5x

Accuracy is unchanged: against an fp32 reference, einsum and F.linear show
identical error (max 0.11420, mean relative 1.76e-04). Results are not
bit-identical to F.linear (different accumulation order), but neither path
is closer to the true value than the other.

Only the single-token case is rerouted. Prefill and the codec model see
multi-token inputs, where rocBLAS's normal GEMM kernels are appropriate,
and fall through untouched.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fast_linear_forward(self, x):
    # Reroute only a true vector input (all leading dims 1) in half precision
    # on GPU; anything else keeps the stock path.
    if (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() == x.shape[-1]
    ):
        out = torch.einsum("ij,j->i", self.weight, x.reshape(-1))
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], out.shape[-1])
    return F.linear(x, self.weight, self.bias)


def apply(model):
    """Swap in the GEMV-friendly forward on every nn.Linear. Returns the count.

    Call after any LoRA merge, so the merged weights are what gets used.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.forward = types.MethodType(_fast_linear_forward, module)
            n += 1
    return n
