cuda_kernel = """
extern "C" __global__
void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index] * input[index];
    }
}
"""

import torch
import torch.utils.cpp_extension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
module = torch.utils.cpp_extension.load_inline(
    name='square',
    cpp_sources='',
    cuda_sources=cuda_kernel,
    functions=['square_kernel']
)

def square(input):
    output = torch.empty_like(input)
    threads_per_block = 1024
    blocks_per_grid = (input.numel() + (threads_per_block - 1)) // threads_per_block
    module.square_kernel(blocks_per_grid, threads_per_block, input, output, input.numel())
    return output

# Example usage
input_tensor = torch.randn(100, device=device)
output_tensor = square(input_tensor)
