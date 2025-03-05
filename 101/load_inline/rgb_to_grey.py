import torch
import torch.cuda as cuda

# CUDA Kernel for RGB to Grayscale Conversion
cuda_source = """
extern "C" {
__global__ void rgb_to_grey(const float* input, float* output, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = h * w;

    if (idx < num_pixels) {
        float r = input[idx] * 0.2229f;
        float g = input[idx + num_pixels] * 0.5879f;
        float b = input[idx + 2 * num_pixels] * 0.1123f;
        output[idx] = r + g + b;
    }
}
}
// c++ wrapper code
torch::Tensor rgb2grey(torch::Tensor input) {
    auto h = input.size(1);
    auto w = input.size(2);
    auto output = torch::empty({h, w}, input.options());
    // call kernel function
    int threads = 256;
    auto num_pixels = h * w;
    int blocks = (num_pixels + threads - 1) / threads;
    rgb_to_grey<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), h, w);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;

}
"""
cpp_wrapper="torch::Tensor rgb2grey(torch::Tensor input);"
# 编译 CUDA 代码
from torch.utils.cpp_extension import load_inline
module = load_inline(name="rgb_to_grey_cuda", cpp_sources=cpp_wrapper, cuda_sources=cuda_source, functions=["rgb2grey"])

# PyTorch CUDA 调用封装
def rgbToGrey_cuda(x):
    c, h, w = x.shape
    assert c == 3, "Input must be a 3-channel RGB image"
    
    # num_pixels = h * w
    # x_flat = x.contiguous().view(-1)
    
    res = module.rgb2grey(x.contiguous())
    
    return res.view(h, w)

# 测试代码
x = torch.randn(3, 512, 512, device="cuda", dtype=torch.float32)  # 512x512 RGB 随机图像
grey = rgbToGrey_cuda(x)
print(grey.shape)  # 应该输出 (512, 512)
