
cuda_source=r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

int cdiv(int num, int div) {
  return ((num + div - 1) / div);
}

// input and output are uint8 type, n is num of pixels
__global__ void rgb_to_grey_cuda(unsigned char* input, unsigned char* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  output[i] = 0.2 * input[i] + 0.5 * input[i + n] + 0.1 * input[i+2*n];
}

// c++ wrapper and kernel setup and prepare input
torch::Tensor rgb_to_grey(torch::Tensor input) {
  //auto (c, h w) = input.size();
  int c = input.size(0);
  int h = input.size(1);
  int w = input.size(2);
  
  auto input_c = input.contiguous();

  auto threads_per_block = 256;
  auto blocks = cdiv(h * w, threads_per_block);

  auto output = torch::empty({h, w}, input.options());

  rgb_to_grey_cuda<<<blocks, threads_per_block>>>(input_c.data_ptr<unsigned char>(),
                          output.data_ptr<unsigned char>(),
                          h * w);
  return output;                        
}
'''
cpp_source = r"torch::Tensor rgb_to_grey(torch::Tensor input);"

from torch.utils.cpp_extension import load_inline

load_inline(
    name="my_module",
    functions=['rgb_to_grey'],
    cpp_sources=[cpp_source,],
    cuda_sources=[cuda_source,],
    extra_cflags=['-O2'],
)

import torch
import my_module

input_tensor = torch.randint(0, 255, (3, 1000, 1000), dtype=torch.uint8, device='cuda')
output_tensor = my_module.rgb_to_grey(input_tensor).cpu()

import matplotlib.pyplot as plt

plt.imshow(output_tensor.numpy(), cmap='gray')

with torch.profiler.profile() as prof:
    for i in range(10_000):
        my_module.rgb_to_grey(input_tensor)
        torch.cuda.synchronize()

print(prof.key_averages().table())