import triton
import torch
import triton.language as tl
from torch._inductor import triton_helpers
from torch._inductor.triton_heuristics import grid

@triton.jit
def pointwise_add_relu_fusion_512(in_out_ptr0, in_ptr0, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # dense @ weights
    x2 = xindex
    # bias
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    # bias + dense @ weights
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)


if __name__ == '__main__':
    torch.cuda.set_device(0)  # no-op to ensure context
    X = torch.ones(size=(128, 512), device='cuda')
    print(X[:3, :3])
    Y = torch.ones(size=(512,), device='cuda')
    print(Y[:3])
    eager_result = torch.maximum(X + Y, torch.tensor(0., device='cuda'))
    print(eager_result[:3, :3])
    pointwise_add_relu_fusion_512[grid(65536)](X, Y, 512)
    print(X)
    torch.testing.assert_close(X, eager_result, rtol=1e-4, atol=1e-4)
