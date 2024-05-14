
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


cpp_fused_add_mul_relu_0 = async_compile.cpp('''
#include "/tmp/torchinductor_ksharma/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 * tmp3;
            auto tmp5 = tmp0 + tmp4;
            auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_sigmoid_1 = async_compile.cpp('''
#include "/tmp/torchinductor_ksharma/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 * tmp3;
            auto tmp5 = tmp0 + tmp4;
            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr0[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
            auto tmp4 = decltype(tmp0)(tmp0 + tmp3);
            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (16, 5), (5, 1))
    assert_size_stride(primals_2, (5, 8), (8, 1))
    assert_size_stride(primals_3, (8, 5), (5, 1))
    assert_size_stride(primals_4, (5, 4), (4, 1))
    assert_size_stride(primals_5, (8, 16), (16, 1))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (4, 8), (8, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (7, 16), (16, 1))
    buf0 = empty((7, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__self___layers___0___linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(reinterpret_tensor(primals_6, (7, 8), (0, 1), 0), primals_9, reinterpret_tensor(primals_5, (16, 8), (1, 16), 0), alpha=1, beta=1, out=buf0)
    del primals_5
    del primals_6
    buf1 = empty((7, 5), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.mm]
    extern_kernels.mm(primals_9, primals_1, out=buf1)
    del primals_1
    buf2 = empty((7, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf1, primals_2, out=buf2)
    buf3 = buf0; del buf0  # reuse
    cpp_fused_add_mul_relu_0(c_void_p(buf3.data_ptr()), c_void_p(buf2.data_ptr()))
    del buf2
    buf4 = empty((7, 4), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__self___layers___2___linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(reinterpret_tensor(primals_8, (7, 4), (0, 1), 0), buf3, reinterpret_tensor(primals_7, (8, 4), (1, 8), 0), alpha=1, beta=1, out=buf4)
    del primals_8
    buf5 = empty((7, 5), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, primals_3, out=buf5)
    buf6 = empty((7, 4), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, primals_4, out=buf6)
    buf7 = buf4; del buf4  # reuse
    cpp_fused_add_mul_sigmoid_1(c_void_p(buf7.data_ptr()), c_void_p(buf6.data_ptr()))
    return (buf7, primals_9, buf3, buf7, reinterpret_tensor(buf5, (5, 7), (1, 5), 0), reinterpret_tensor(primals_4, (4, 5), (1, 4), 0), reinterpret_tensor(primals_3, (5, 8), (1, 5), 0), reinterpret_tensor(primals_7, (4, 8), (8, 1), 0), reinterpret_tensor(buf1, (5, 7), (1, 5), 0), reinterpret_tensor(primals_2, (8, 5), (1, 8), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 5), (5, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((5, 8), (8, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((8, 5), (5, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((5, 4), (4, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((8, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((4, 8), (8, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((7, 16), (16, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
