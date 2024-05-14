#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fused_kernels_lora_on_mlp.cuh"


__global__ void fused_add_mul_relu_kernel(float *dense_in_out_ptr,
                                          const float *scalar_ptr,
                                          const float *dense_ptr,
                                          const int num_weights,
                                          const int xnumel,
                                          const double multiplier) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < xnumel) {
        int scalar_index = index % num_weights;
        float tmp0 = dense_in_out_ptr[index];
        float tmp1 = scalar_ptr[scalar_index];
        float tmp3 = dense_ptr[index];
        float ma_result = max(0.0f, multiplier * tmp3 + tmp0 + tmp1);
        dense_in_out_ptr[index] = ma_result;
    }
}

torch::Tensor fused_add_mul_relu(torch::Tensor in_out,
                                 const torch::Tensor &bias,
                                 const torch::Tensor &in,
                                 const double multiplier) {
    auto numel = in_out.numel();
    auto sizes = in_out.sizes();
    const int XBLOCK = sizes[0];
    dim3 threadsPerBlock(sizes[1]);
    dim3 numBlocks((numel + XBLOCK - 1) / XBLOCK);
    fused_add_mul_relu_kernel<<<numBlocks, threadsPerBlock>>>(
            in_out.data_ptr<float>(),
            bias.data_ptr<float>(),
            in.data_ptr<float>(),
            sizes[1],
            numel,
            multiplier);
    cudaDeviceSynchronize();
    return std::move(in_out);
}