__global__ void matrixMulSimple(float* A, float* B, float* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < width) {
        float value = 0;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

#define TILE_WIDTH 16

__global__ void matrixMulTiled(float* A, float* B, float* C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float value = 0;
    for (int m = 0; m < width/TILE_WIDTH; ++m) {
        As[ty][tx] = A[Row*width + (m*TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m*TILE_WIDTH + ty)*width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if(Row < width && Col < width) {
        C[Row*width + Col] = value;
    }
}
