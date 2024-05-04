# Look at this test for inspiration
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py

import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper
cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
    }
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# Load the CUDA kernel as a PyTorch extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))

# (cudamode) ubuntu@ip-172-31-9-217:~/cudamode/cudamodelecture1$ python load_inline.py 
# tensor([[ 1.,  4.,  9.],
#         [16., 25., 36.]], device='cuda:0')


## No great interaction with ncu

# (cudamode) ubuntu@ip-172-31-9-217:~/cudamode/cudamodelecture1$ ncu python load_inline.py 
# ==PROF== Connected to process 55916 (/opt/conda/envs/cudamode/bin/python3.10)
# /opt/conda/envs/cudamode/lib/python3.10/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 36: API call is not supported in the installed CUDA driver (Triggered internally at /opt/conda/conda-bld/pytorch_1702400410390/work/c10/cuda/CUDAFunctions.cpp:108.)
#   return torch._C._cuda_getDeviceCount() > 0
# No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
# Traceback (most recent call last):
#   File "/home/ubuntu/cudamode/cudamodelecture1/load_inline.py", line 7, in <module>
#     a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
#   File "/opt/conda/envs/cudamode/lib/python3.10/site-packages/torch/cuda/__init__.py", line 298, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 36: API call is not supported in the installed CUDA driver
# ==PROF== Disconnected from process 55916
# ==ERROR== The application returned an error code (1).
# ==WARNING== No kernels were profiled.
# ==WARNING== Profiling kernels launched by child processes requires the --target-processes all option.