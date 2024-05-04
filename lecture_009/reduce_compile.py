# TORCH_LOGS="output_code" python reduce_compile.py
import torch 

@torch.compile
def f(a):
    c = torch.sum(a)
    return c

f(torch.randn(10).cuda())
