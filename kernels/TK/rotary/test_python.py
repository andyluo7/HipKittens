import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange, repeat
import tk_kernel

B = 1
H = 1
N = 64
D = 64

D_2 = D // 2

torch.random.manual_seed(42)
x = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()

def get_output(x, rotary_emb_base=10000, rotary_emb_dim=D, dtype=torch.bfloat16):

    t = torch.arange(N, device=x.device, dtype=dtype) # We want fp32 here
    inv_freq = 1.0 / (rotary_emb_base ** (torch.arange(0, rotary_emb_dim, 2, device=x.device, dtype=dtype) / rotary_emb_dim))
    freqs = torch.outer(t, inv_freq).to(dtype=dtype)
    cos_in = torch.cos(freqs).to(dtype=dtype)
    sin_in = torch.sin(freqs).to(dtype=dtype)

    ro_dim = cos_in.shape[-1] * 2
    assert ro_dim <= x.shape[-1] 

    ###

    x_ro_dim     = x[..., :ro_dim]
    x_ro_dim_end = x[..., ro_dim:]

    x1, x2 = x_ro_dim.chunk(2, dim=-1)              # D/2, D/2
    rotated_x = torch.cat((-x2, x1), dim=-1)        # D
    
    cos = repeat(cos_in, "n d -> 1 n (2 d)" )
    sin = repeat(sin_in, "n d -> 1 n (2 d)" )
    o = torch.cat([x_ro_dim * cos + rotated_x * sin, x_ro_dim_end], dim=-1)

    return o, ro_dim, cos_in, sin_in

o, ro_dim, cos_in, sin_in = get_output(x)

o_tk = torch.zeros_like(o).bfloat16()
sin_tk = sin_in.to(torch.bfloat16).cuda()
cos_tk = cos_in.to(torch.bfloat16).cuda()
tk_kernel.dispatch_rotary(x, o_tk, sin_tk, cos_tk)

# Compare
o_diff = o - o_tk
print("o: ", o[0, 0, 0, :8])
print("o_tk: ", o_tk[0, 0, 0, :8])
print("o_diff: ", o_diff[0, 0, 0, :8])

max_diff = o_diff.abs().max().item()
print("max_diff: ", max_diff)
    
# breakpoint()


