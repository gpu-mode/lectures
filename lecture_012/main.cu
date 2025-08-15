#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Declare your kernels (must match your .cu definitions)
extern "C" __global__ void flash_attention_spilling_from_registers_k(
    float *out, 
    float *out_l, 
    float *Q,
    float *K, 
    float *V, 
    float scaling, 
    int n, 
    int T_r, 
    int T_c
);

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

int main() {
    // Problem sizes (small enough to test, big enough for profiling)
    int n = 128;       // number of rows in Q
    int n_inp = 128;   // number of rows in K and V
    int d = 64;        // feature dimension

    int B_r = 16;      // block tile rows
    int B_c = 16;      // block tile cols
    int block_dim_x = 16;
    int block_dim_y = 16;

    // Host allocations
    size_t size_Q = n * d * sizeof(float);
    size_t size_K = n_inp * d * sizeof(float);
    size_t size_V = n_inp * d * sizeof(float);
    size_t size_out = n * d * sizeof(float);
    size_t size_out_l = n * sizeof(float);

    float *h_Q = new float[n * d];
    float *h_K = new float[n_inp * d];
    float *h_V = new float[n_inp * d];

    // Initialize with some data
    for (int i = 0; i < n * d; i++) h_Q[i] = static_cast<float>(i % 13) * 0.1f;
    for (int i = 0; i < n_inp * d; i++) {
        h_K[i] = static_cast<float>((i % 7) * 0.2f);
        h_V[i] = static_cast<float>((i % 5) * 0.3f);
    }

    // Device allocations
    float *d_Q, *d_K, *d_V, *d_out, *d_out_l;
    CUDA_CHECK(cudaMalloc(&d_Q, size_Q));
    CUDA_CHECK(cudaMalloc(&d_K, size_K));
    CUDA_CHECK(cudaMalloc(&d_V, size_V));
    CUDA_CHECK(cudaMalloc(&d_out, size_out));
    CUDA_CHECK(cudaMalloc(&d_out_l, size_out_l));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice));

    float scaling = 1.0f / sqrtf(static_cast<float>(d));
    int T_r = cdiv(n, B_r);
    int T_c = cdiv(n_inp, B_c);

    dim3 blocks(1, 1);
    dim3 threads(block_dim_x, block_dim_y);

    printf("Launching kernel...\n");

    // Kernel launch
    flash_attention_spilling_from_registers_k<<<blocks, threads>>>(
        d_out, 
        d_out_l,
        d_Q, 
        d_K, 
        d_V, 
        scaling,
        n,
        T_r,
        T_c
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel finished.\n");

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_out_l));

    return 0;
}
