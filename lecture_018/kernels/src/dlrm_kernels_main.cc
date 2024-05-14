#include <iostream>
#include "pointwise_add_relu_fused.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>


int main()
{
    torch::manual_seed(0);
    std::vector<int64_t> sizes = {8, 10};
    auto x = torch::randn(sizes, torch::kCUDA);
    auto bias = torch::randn(sizes[1], torch::kCUDA);
    std::cout << "Tensor x:\n" << x << '\n';
    std::cout << "Tensor y:\n" << bias << '\n';
    auto expected_result = torch::clamp_min(x + bias, 0.0);
    std::cout << "Expected:\n" << expected_result << '\n';
    auto result = add_relu_fusion(x, bias);
    std::cout << "Result:\n" << result << '\n';
    std::cout << "All Match: " << (torch::allclose(expected_result, result) ? "true" : "false") << '\n';
    return 0;
}
