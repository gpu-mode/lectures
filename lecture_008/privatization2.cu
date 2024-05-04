#include <stdio.h>
#include <cuda_runtime.h>

// Kernel without privatization: Direct global memory access
__global__ void windowSumDirect(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = idx + i;
            if (accessIdx >= 0 && accessIdx < n) {
                sum += input[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

// Kernel with privatization: Preload window elements into registers
__global__ void windowSumPrivatized(const float *input, float *output, int n, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWindow = windowSize / 2;
    __shared__ float sharedData[1024]; // Assuming blockDim.x <= 1024

    // Load input into shared memory (for demonstration, assuming window fits into shared memory)
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
        __syncthreads(); // Ensure all loads are complete

        float sum = 0.0f;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            int accessIdx = threadIdx.x + i;
            // Check bounds within shared memory
            if (accessIdx >= 0 && accessIdx < blockDim.x && (idx + i) < n && (idx + i) >= 0) {
                sum += sharedData[accessIdx];
            }
        }
        output[idx] = sum;
    }
}

void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1.0f; // Simple initialization for demonstration
    }
}

int main() {
    int n = 1<<20; // Example array size
    int windowSize = 5; // Example window size
    float *input, *output;
    float *d_input, *d_output;

    input = (float*)malloc(n * sizeof(float));
    output = (float*)malloc(n * sizeof(float));

    // Initialize input array
    initializeArray(input, n);

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernels
    windowSumDirect<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    windowSumPrivatized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, windowSize);
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost); // Copy result back for verification

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}
