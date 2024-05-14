from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_mul_relu(in_out_ptr0, in_ptr0, in_ptr1, xnumel, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
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
    tmp7 = tl.maximum(0, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)


@triton.jit
def fused_add_mul_relu_cleaner(dense_in_out_ptr, scalar_ptr, dense_ptr, num_weights, xnumel, multiplier,
                               BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    scalar_index = index % num_weights
    tmp0 = tl.load(dense_in_out_ptr + index, mask)
    tmp1 = tl.load(scalar_ptr + scalar_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(dense_ptr + index, mask)
    # later found that there is a tl.fma function that can be used to do the fused multiply add
    # https://triton-lang.org/main/python-api/generated/triton.language.fma.html#triton.language.fma
    # Option 1
    ma_result = tl.maximum(0, multiplier * tmp3 + tmp0 + tmp1)
    # Option 2
    # ma_result = tl.maximum(0, tl.math.fma(multiplier, tmp3, tmp0) + tmp1)
    tl.store(dense_in_out_ptr + index, ma_result, mask)


def fused_add_mul_relu_torch(in_out_tensor: torch.Tensor, bias: torch.Tensor, in_tensor: torch.Tensor) -> torch.Tensor:
    # print("calling fused_add_mul_relu_torch")
    grid = lambda meta: (triton.cdiv(in_out_tensor.numel(), meta['BLOCK_SIZE']),)
    BLOCK_SIZE = min(1024, in_out_tensor.numel())
    fused_add_mul_relu[grid](in_out_tensor, bias, in_tensor, in_out_tensor.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return in_out_tensor


def fused_add_mul_relu_cleaner_torch(in_out_tensor: torch.Tensor, bias: torch.Tensor,
                                     in_tensor: torch.Tensor) -> torch.Tensor:
    # print("calling fused_add_mul_relu_torch")
    grid = lambda meta: (triton.cdiv(in_out_tensor.numel(), meta['BLOCK_SIZE']),)
    BLOCK_SIZE = min(1024, in_out_tensor.numel())
    num_weights = bias.numel()
    fused_add_mul_relu_cleaner[grid](
        in_out_tensor, bias, in_tensor, num_weights, in_out_tensor.numel(), multiplier=0.5, BLOCK_SIZE=BLOCK_SIZE)
    return in_out_tensor


def get_inputs(batch_size: int = 8, weight_size: int = 8, add_manual_seed: bool = False):
    if add_manual_seed:
        torch.manual_seed(0)
    dense_size = (batch_size, weight_size)
    in_out_tensor = torch.randn(dense_size, device='cuda', dtype=torch.float32)
    in_tensor = torch.randn(dense_size, device='cuda', dtype=torch.float32)
    bias = torch.randn((1, weight_size), device='cuda', dtype=torch.float32)
    return in_out_tensor, in_tensor, bias


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size', 'weight_size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[(2 ** i, 2 ** j) for i, j in zip(range(2, 20, 2), range(2, 11, 1))],
        # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch.compile.generated', 'cleaner'],  # Possible values for `line_arg`.
        line_names=['torch.compile.generated', 'cleaner'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(batch_size, weight_size, provider):
    in_out_tensor, in_tensor, bias = get_inputs(batch_size=batch_size, weight_size=weight_size, add_manual_seed=True)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch.compile.generated":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_add_mul_relu_torch(in_out_tensor, bias, in_tensor),
                                                     quantiles=quantiles)
    if provider == "cleaner":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_add_mul_relu_cleaner_torch(in_out_tensor, bias, in_tensor), quantiles=quantiles)
    gbps = lambda ms: 12 * (batch_size * weight_size) / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    print(triton.__version__)
    in_out_tensor, in_tensor, bias = get_inputs(add_manual_seed=True)
    expected_output = torch.maximum(in_out_tensor + 0.5 * in_tensor + bias, torch.tensor(0., device='cuda'))
    print("Input", in_out_tensor)
    print("Expected Output", expected_output)
    BLOCK_SIZE = 8
    grid = lambda meta: (triton.cdiv(in_out_tensor.numel(), meta['BLOCK_SIZE']),)
    fused_add_mul_relu[grid](in_out_tensor, bias, in_tensor, in_out_tensor.numel(), BLOCK_SIZE=BLOCK_SIZE)
    print("Output 1", in_out_tensor)

    torch.testing.assert_close(in_out_tensor, expected_output, rtol=1e-4, atol=1e-4)

    in_out_tensor, in_tensor, bias = get_inputs(add_manual_seed=True)
    num_weights = bias.numel()
    fused_add_mul_relu_cleaner[grid](in_out_tensor, bias, in_tensor, num_weights, in_out_tensor.numel(), multiplier=0.5,
                                     BLOCK_SIZE=BLOCK_SIZE)
    print("Output 2", in_out_tensor)
    torch.testing.assert_close(in_out_tensor, expected_output, rtol=1e-4, atol=1e-4)

    benchmark.run(print_data=True, show_plots=True)
