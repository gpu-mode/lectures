#include <stdio.h>
#include <cuda_runtime.h>

__global__ void processArrayWithDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx] % 2 == 0) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] + 1;
        }
    }
}

__global__ void processArrayWithoutDivergence(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int isEven = !(data[idx] % 2);
        data[idx] = isEven * (data[idx] * 2) + (!isEven) * (data[idx] + 1);
    }
}

void benchmarkKernel(void (*kernel)(int *, int), int *data, int N, const char *kernelName) {
    int *devData;
    cudaMalloc(&devData, N * sizeof(int));
    cudaMemcpy(devData, data, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(devData, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s took %f milliseconds\n", kernelName, milliseconds);

    cudaMemcpy(data, devData, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 1 << 20; // Example size
    int *data = (int *)malloc(N * sizeof(int));

    // Initialize data
    for(int i = 0; i < N; i++) {
        data[i] = i;
    }

    benchmarkKernel(processArrayWithDivergence, data, N, "processArrayWithDivergence");
    benchmarkKernel(processArrayWithoutDivergence, data, N, "processArrayWithoutDivergence");

    free(data);
    return 0;
}
