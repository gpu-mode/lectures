#include <iostream>
#include "fused_kernels_lora_on_mlp.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>


int main()
{
    torch::manual_seed(0);
    std::vector<int64_t> sizes = {8, 10};
    auto x = torch::randn(sizes, torch::kCUDA);
    auto bias = torch::randn(sizes[1], torch::kCUDA);
    auto y = torch::randn(sizes, torch::kCUDA);
    const float mult = 2.3;
    std::cout << "Tensor x:\n" << x << '\n';
    std::cout << "Tensor bias:\n" << bias << '\n';
    std::cout << "Tensor y:\n" << y << '\n';
    auto expected_result = torch::clamp_min(x + bias + y * mult, 0.0);
    std::cout << "Expected:\n" << expected_result << '\n';
    auto result = fused_add_mul_relu(x, bias, y, mult);
    std::cout << "Result:\n" << result << '\n';
    std::cout << "All Match: " << (torch::allclose(expected_result, result) ? "true" : "false") << '\n';
    return 0;
}
