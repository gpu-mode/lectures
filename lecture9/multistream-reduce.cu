#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2
#define NUM_DEVICES 2

__global__ void CoarsenedReduction(float* input, float* output, int size) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int i = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0.0f;

    // Reduce within a thread
    for (unsigned int tile = 0; tile < COARSE_FACTOR; ++tile) {
        unsigned int index = i + tile * blockDim.x;
        if (index < size) {
            sum += input[index];
        }
    }

    input_s[t] = sum;
    __syncthreads();

    // Reduce within a block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Reduce over blocks
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    const int size = 10000;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;  // Example: Initialize all elements to 1
    }

    // Create CUDA streams for pipelining
    cudaStream_t streams[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // Allocate memory for input and output on each device
    float* d_input[NUM_DEVICES];
    float* d_output[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_input[i], bytes);
        cudaMalloc(&d_output[i], sizeof(float));
        cudaMemset(d_output[i], 0, sizeof(float));  // Initialize output to 0
    }

    // Copy data from host to each device
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_input[i], h_input, bytes, cudaMemcpyHostToDevice, streams[i]);
    }

    // Launch the kernel with coarsening on each device
    int numBlocks = (size + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        CoarsenedReduction<<<numBlocks, BLOCK_DIM, 0, streams[i]>>>(d_input[i], d_output[i], size);
    }

    // Copy results back to host from each device
    float* d_output_host[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaMallocHost(&d_output_host[i], sizeof(float));
        cudaSetDevice(i);
        cudaMemcpyAsync(d_output_host[i], d_output[i], sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to complete
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Sum the results from each device on the host
    float final_sum = 0.0f;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        final_sum += *d_output_host[i];
    }

    // Print the result
    std::cout << "Sum is " << final_sum << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(d_output_host[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
