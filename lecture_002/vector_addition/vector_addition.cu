#include <cuda.h>
#include <stdio.h>

// compute vector sum C = A + B
// each thread peforms one pair-wise addition
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

void vecAdd(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  size_t size = n * sizeof(float); // sizeof是以字节为单位的，对于float来说是4个字节，对应的是32位。

  // 首先是进行内存的分配
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  // 将数据copy到GPU上
  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  // 给的线程数是256
  const unsigned int numThreads = 256;
  // block 块数位100/256=4 , 4个block块，每个block块有256个线程
  unsigned int numBlocks = cdiv(n, numThreads);

  //执行加法的kernel
  vecAddKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  // 这里计算两个向量的加法
  const int n = 1000;
  float A[n];
  float B[n];
  float C[n];

  // generate some dummy vectors to add
  //生成一些用于添加的虚拟向量
  for (int i = 0; i < n; i += 1) {
    A[i] = float(i);
    B[i] = A[i] / 1000.0f;
  }

  // 调用add的kernel
  vecAdd(A, B, C, n);

  // print result
  for (int i = 0; i < n; i += 1) {
    if (i > 0) {
      printf(", ");
      if (i % 10 == 0) {
        printf("\n");
      }
    }
    printf("%8.3f", C[i]);
  }
  printf("\n");
  return 0;
}
