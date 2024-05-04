#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition without privatization
__global__ void vectorAdd(const float *a, const float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        result[index] = a[index] + b[index];
    }
}

// CUDA kernel for vector addition with privatization
__global__ void vectorAddPrivatized(const float *a, const float *b, float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        float a_private = a[index]; // Load into private memory
        float b_private = b[index]; // Load into private memory
        result[index] = a_private + b_private;
    }
}

// Function to initialize the vectors with dummy data
void initData(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }
}

int main() {
    int n = 1<<20; // Size of the vectors
    float *a, *b, *result, *d_a, *d_b, *d_result;

    // Allocate memory on the host
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    result = (float*)malloc(n * sizeof(float));

    // Initialize vectors
    initData(a, n);
    initData(b, n);

    // Allocate memory on the device
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector addition kernel without privatization
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

    // Copy result back to host
    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the vector addition kernel with privatization
    vectorAddPrivatized<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

    // Copy result back to host
    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(a);
    free(b);
    free(result);

    return 0;
}
