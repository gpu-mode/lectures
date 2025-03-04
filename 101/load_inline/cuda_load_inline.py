import torch
from torch.utils.cpp_extension import load_inline
# Define the CUDA kernel and C++ wrapper
build_directory = './load_inline_cuda'

cuda_source = """
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

    const dim3 threads_per_block(16, 16);
    const dim3 number_of_blocks(
                                (width + threads_per_block.x - 1) / threads_per_block.x,
                                (height + threads_per_block.y - 1) / threads_per_block.y
                                );
    square_matrix_kernel<<<number_of_blocks, threads_per_block>>> (
                                                                matrix.data_ptr<float>(),
                                                                result.data_ptr<float>(),
                                                                width,
                                                                height
                                                                );
    return result;
}
"""
cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# load the inline CUDA extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cuda_sources=cuda_source,
    cpp_sources=cpp_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=['-O2'],
    build_directory=build_directory
)

# test the extension

def test_square_matrix():
    matrix = torch.tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=torch.float32, device='cuda')
    result = square_matrix_extension.square_matrix(matrix)
    assert torch.allclose(result, torch.tensor([[1,4,9], [16,25,36], [49,64,81]], dtype=torch.float32, device='cuda'))
    print(result)



print("Testing square_matrix extension...")
test_square_matrix()
print("All tests passed!")