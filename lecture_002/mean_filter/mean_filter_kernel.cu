#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    /*
    均值滤波是一种简单的图像模糊技术，通过计算每个像素周围邻域内所有像素值的平均值来更新像素值，从而减少图像噪声。
    */
    // 计算当前线程处理的列位置，blockIdx.x 和 blockDim.x 确定线程块和线程在x轴上的位置。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算当前线程处理的行位置，blockIdx.y 和 blockDim.y 确定线程块和线程在y轴上的位置。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算当前线程处理的颜色通道，threadIdx.z 确定线程在z轴上的位置。
    int channel = threadIdx.z;

    // 计算当前通道的基地址偏移量，用于访问多通道图像数据。
    int baseOffset = channel * height * width;
    if (col < width && row < height) {

        // 初始化当前像素的累积像素值和像素计数。
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-radius; blurRow <= radius; blurRow += 1) {
            for (int blurCol=-radius; blurCol <= radius; blurCol += 1) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }

        // 计算邻域内所有像素的平均值，并将其赋值给输出图像的对应像素。
        // 这里使用了类型转换，因为除法结果可能不是整数。
        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}


// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor mean_filter(torch::Tensor image, int radius) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16, channels); // 这是一个三维的

    /*
    每个线程块应该处理多少个像素列。即是每个元素对应一个线程，就是一个线程处理一个像素点
    */
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),// 宽的blocks为width/threads_per_block.x，threads_per_block.x的总是是16
        cdiv(height, threads_per_block.y) // 高的blocks为height/threads_per_block.y, threads_per_block.y的总是为16
    );

    mean_filter_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
