#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // 这里考虑二维矩阵的存储方式，记住，在计算机中，所有的都是从0开始计数的。
        // 当这个row是0的时候是第0行，所以这个offset就是0+列，0+col，这个是只存一个灰度的图像值，所以输出，不用乘channel数。
        int outputOffset = row * width + col;
        // 由于输入是RGB的三通道，这里的通道你可以简单的理解为三个矩阵，一个R矩阵，一个G矩阵，一个B矩阵
        //image[height][width][3]; // height 为图像高度，width 为图像宽度在这个数组中，
        // image[y][x][0] 将访问像素点 (x, y) 的红色分量，image[y][x][1] 访问绿色通道，image[y][x][2] 访问蓝色分量。

        // 还有一种简单的理解方法，当第一个线程的时候(row * width + col) 这个是0，那么inputOffset就是0，接下来取RGB的值就是
        // 取连续的值，
        /* 假设我现在已经找到了(row * width + col) 
        [0,1,2,  // 这是一个RGB
        3,4,5,   // 这是一个RGB
        6,7,8,   // 这是一个RGB
        9,0，0]  // 这是一个RGB
        */
        int inputOffset = (row * width + col) * channels;

        unsigned char r = input[inputOffset + 0];   // red
        unsigned char g = input[inputOffset + 1];   // green
        unsigned char b = input[inputOffset + 2];   // blue

        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}


// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block 这里使用的是256个线程
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));  // block数等于线程数/shape的大小，shape的每个维度都有对应的block，width/threads_per_block.x,表示每个线程块要处理多少元素。

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
