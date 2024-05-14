
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_ksharma/q7/cq7h4dlcv3kttko5qpzbdwnfoxz5v2zfabffxtrfd77uay2lripi.py
# Source Nodes: [add, l__self___layers_1, x], Original ATen: [aten.add, aten.mul, aten.relu]
# add => add
# l__self___layers_1 => relu
# x => mul
triton_poi_fused_add_mul_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = triton_helpers.maximum(0, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ksharma/kl/cklo4lcv3wxgk3s75u4iojwegc57tuf6phwfqmtok3kndqjho6hh.py
# Source Nodes: [add_1, l__self___layers_3, x_1], Original ATen: [aten.add, aten.mul, aten.sigmoid]
# add_1 => add_1
# l__self___layers_3 => sigmoid
# x_1 => mul_1
triton_poi_fused_add_mul_sigmoid_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (8, 16), (16, 1))
    assert_size_stride(primals_2, (8, ), (1, ))
    assert_size_stride(primals_3, (4, 8), (8, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (16, 5), (5, 1))
    assert_size_stride(primals_6, (5, 8), (8, 1))
    assert_size_stride(primals_7, (8, 5), (5, 1))
    assert_size_stride(primals_8, (5, 4), (4, 1))
    assert_size_stride(primals_9, (7, 16), (16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((7, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        # X1 <- X @ W1 [7 X 8]
        extern_kernels.mm(primals_9, reinterpret_tensor(primals_1, (16, 8), (1, 16), 0), out=buf0)
        del primals_1
        buf1 = empty((7, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.mm]
        # xa1 <- X @ A1 [7 X 5]
        extern_kernels.mm(primals_9, primals_5, out=buf1)
        del primals_5
        buf2 = empty((7, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.mm]
        # xa2 <- xa1 @ B1 [7 X 8] <- 7 X 5 @ 5 X 8
        extern_kernels.mm(buf1, primals_6, out=buf2)
        buf3 = buf0; del buf0  # reuse
        # Source Nodes: [add, l__self___layers_1, x], Original ATen: [aten.add, aten.mul, aten.relu]
        stream0 = get_cuda_stream(0)
        # out1 = relu(add(xa2, X1)) [7 X 8]
        triton_poi_fused_add_mul_relu_0.run(buf3, primals_2, buf2, 56, grid=grid(56), stream=stream0)
        del buf2
        del primals_2
        buf4 = empty((7, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf3, reinterpret_tensor(primals_3, (8, 4), (1, 8), 0), out=buf4)
        buf5 = empty((7, 5), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, primals_7, out=buf5)
        buf6 = empty((7, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, primals_8, out=buf6)
        buf7 = buf4; del buf4  # reuse
        # Source Nodes: [add_1, l__self___layers_3, x_1], Original ATen: [aten.add, aten.mul, aten.sigmoid]
        triton_poi_fused_add_mul_sigmoid_1.run(buf7, primals_4, buf6, 28, grid=grid(28), stream=stream0)
        del buf6
        del primals_4
        return (buf7, primals_9, buf3, buf7, reinterpret_tensor(buf5, (5, 7), (1, 5), 0), reinterpret_tensor(primals_8, (4, 5), (1, 4), 0), reinterpret_tensor(primals_7, (5, 8), (1, 5), 0), reinterpret_tensor(primals_3, (4, 8), (8, 1), 0), reinterpret_tensor(buf1, (5, 7), (1, 5), 0), reinterpret_tensor(primals_6, (8, 5), (1, 8), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((5, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((5, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((7, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
