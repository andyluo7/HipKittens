import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

random.seed(0)
torch.manual_seed(0)

b = 1
h = 1
n = 32
d = 32

# pytorch
x_a = torch.randn((b, h, n, d), dtype=torch.bfloat16, device='cuda')
x_b = torch.randn((b, h, n, d), dtype=torch.bfloat16, device='cuda')
x_accum = torch.zeros((b, h, n, d), dtype=torch.float32, device='cuda')
# y = x.sum(dim=-1, keepdim=True)
y = torch.matmul(x_a, x_b.transpose(-2, -1)).float()
# y = x_b

# tk
y_tk = torch.zeros_like(y).float()
tk_kernel.dispatch_micro(x_a, x_b, y_tk, x_accum)

print(y.shape, y_tk.shape)

# check
diff = (y - y_tk).abs()
max_diff = diff.max()
print(y.shape,  x_a.shape, x_b.shape)


# print(y[0, 0, 0:2, :8])
print(diff[0, 0, 0:, :])
num_diff = (diff > 1e-3).sum()
print(f"num_diff: {num_diff} / {diff.numel()}")
# print()

# print(x_b[0, 0, 0:, :].sum(dim=-1))
# print(y_tk[0, 0, 0:1, :])
# print(y[0, 0, 0:1, :])

# print(x_b[0, 0, 0:16, :])
# print(y_tk[0, 0, 0:16, :])
# print(y[0, 0, 0:16, :])

# print(y[0, 0, 8:10, :8])
# print(y_tk[0, 0, 8:10, :8])
print(f"diff: {max_diff}")




