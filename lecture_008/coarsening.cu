#include <stdio.h>

#define N 1024
#define THREADS_PER_BLOCK 256 // This is just an example block size

// Original vector addition kernel without coarsening
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Vector addition kernel with thread coarsening
// Assuming a coarsening factor of 2
__global__ void VecAddCoarsened(float* A, float* B, float* C)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Coarsening factor applied here
    if (i < N)
        C[i] = A[i] + B[i];
    if (i + 1 < N) // Handle the additional element due to coarsening
        C[i + 1] = A[i + 1] + B[i + 1];
}

void random_init(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(float);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (float *)malloc(size); random_init(a, N);
    b = (float *)malloc(size); random_init(b, N);
    c = (float *)malloc(size);

    cudaEvent_t start, stop, startCoarsened, stopCoarsened;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCoarsened);
    cudaEventCreate(&stopCoarsened);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // warmup
    VecAdd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1) / (2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Start timer for VecAdd kernel
    cudaEventRecord(start);
    VecAdd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    // Wait for VecAdd kernel to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("VecAdd execution time: %f ms\n", milliseconds);

    // Start timer for VecAddCoarsened kernel
    cudaEventRecord(startCoarsened);
    VecAddCoarsened<<<(N + 2*THREADS_PER_BLOCK - 1) / (2*THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stopCoarsened);

    // Wait for VecAddCoarsened kernel to finish
    cudaEventSynchronize(stopCoarsened);

    float millisecondsCoarsened = 0;
    cudaEventElapsedTime(&millisecondsCoarsened, startCoarsened, stopCoarsened);
    printf("VecAddCoarsened execution time: %f ms\n", millisecondsCoarsened);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startCoarsened);
    cudaEventDestroy(stopCoarsened);

    return 0;
}
