import time
import torch

import triton
import triton.language as tl
from triton.tools.disasm import get_sass
from torch.profiler import profile


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == "__main__":
    size = 1024
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    output = torch.empty_like(x)
    compiled_kernel = add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    print(compiled_kernel.asm.keys())
    print(compiled_kernel.asm["ttir"])
    import ipdb; ipdb.set_trace()
    print(get_sass(compiled_kernel.asm["cubin"]))

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs"),
        with_stack=True,
    )
    prof.start()

    for i in range(10):
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
        prof.step()

    prof.stop()
