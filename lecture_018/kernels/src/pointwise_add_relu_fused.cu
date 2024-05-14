#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "pointwise_add_relu_fused.cuh"


__global__ void add_relu_fusion_kernel(float* in_out_ptr0, const float* in_ptr0, const int xnumel ,const int XBLOCK) {
    const int tid = threadIdx.x;
    const int xoffset = blockIdx.x * XBLOCK;
    const int xindex = xoffset + tid;
    const bool xmask = xindex < xnumel;

    if (xmask) {
        int x2 = xindex;
        int x0 = xindex % XBLOCK;
        float tmp0 = in_out_ptr0[x2];
        float tmp1 = in_ptr0[x0];
        float tmp2 = tmp0 + tmp1;
        float tmp3 = max(0.0f, tmp2); // ReLU operation

        in_out_ptr0[x2] = tmp3;
    }
}

torch::Tensor add_relu_fusion(torch::Tensor in_out, const torch::Tensor& in) {
    auto sizes = in_out.sizes();
    auto XBLOCK = sizes[1];
    auto numel = in_out.numel();
    dim3 threadsPerBlock(XBLOCK);
    dim3 numBlocks((numel + XBLOCK - 1) / XBLOCK);
    add_relu_fusion_kernel<<<numBlocks, threadsPerBlock>>>(in_out.data_ptr<float>(), in.data_ptr<float>(), numel, XBLOCK);
    cudaDeviceSynchronize();
    return std::move(in_out);
}