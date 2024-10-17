// Compile me as
// clang++ --std=c++17 gemm_perf_studies.mm -framework Metal -framework
// Foundation Implements matmul of row-first with colum-first matrices using
// naive, vec4 and mat4
#include <Metal/Metal.h>

#include <chrono>
#include <iostream>
#include <stdexcept>

const std::string &naive_gemm = R"METAL(// Naive
// One thread per output element
kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z-1
  constant auto *A_ptr = A + m * lda;
  constant auto *B_ptr = B + n * lda;

  float rc = 0.0;
  for (uint k = 0; k < sizes.y; k++) {
    const auto a_val = A_ptr[k];
    const auto b_val = B_ptr[k];
    rc += a_val * b_val;
  }
  outputData[m * ldc + n] = rc;
}
)METAL";

const std::string &vec4_gemm = R"METAL(// SIMD(vec4)
// One thread per output element
using namespace metal;

kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z-1
  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * lda);
  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * lda);

  float rc = 0.0;
  for (uint k = 0; k < sizes.y / 4; k++) {
    rc += dot(A_ptr[k], B_ptr[k]);
  }
  outputData[m * ldc + n] = rc;
}
)METAL";

const std::string &mat4_gemm = R"METAL(// SIMD(mat4xvec4)
// One thread per group of 4 output elements, 8x8 blocks
using namespace metal;

kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z/4-1
  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * lda);
  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * 4 * lda);

  float4 rc = 0.0;
  for (uint k = 0; k < sizes.y / 4; k++) {
    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {
      b_mat[j] = B_ptr[k + j * lda /4];
    }
    rc += transpose(b_mat) * A_ptr[k];
  }
  reinterpret_cast<device float4*>(outputData + m * ldc)[n] = rc;
}
)METAL";

template <typename Callable>
float measure_time(unsigned repeat_cnt, Callable c) {
  using namespace std::chrono;
  auto start = high_resolution_clock::now();
  for (unsigned idx = 0; idx < repeat_cnt; idx++) {
    c();
  }
  auto end = high_resolution_clock::now();
  return duration<float>(end - start).count() / repeat_cnt;
}
id<MTLDevice> getMetalDevice() {
  NSArray *devices = [MTLCopyAllDevices() autorelease];
  if (devices.count == 0) {
    throw std::runtime_error("Metal is not supported");
  }
  return devices[0];
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    throw std::runtime_error("Can't allocate " + std::to_string(length) +
                             " bytes on GPU");
  }
  return rc;
}

id<MTLLibrary> compileLibraryFromSource(id<MTLDevice> device,
                                        const std::string &source) {
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion3_1];
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:options
                     error:&error];
  if (library == nil) {
    throw std::runtime_error(std::string("Failed to compile: ") +
                             error.description.UTF8String);
  }
  return library;
}

template <unsigned col_div = 1>
void benchmark_gemm(id<MTLDevice> dev, const std::string &shader_source,
                    unsigned M, unsigned N, unsigned K) {
  // Shader name is encoded in the First line of the source skipping comment
  // prefix
  auto shader_name = shader_source.substr(3, shader_source.find('\n') - 3);

  // Load shader code and find gemm function
  auto lib = compileLibraryFromSource(dev, shader_source);
  id<MTLFunction> func = [lib newFunctionWithName:@"gemm"];
  if (func == nil) {
    throw std::runtime_error("Can't get function");
  }
  NSError *error = nil;
  auto cpl = [lib.device newComputePipelineStateWithFunction:func error:&error];
  if (cpl == nil) {
    throw std::runtime_error(
        std::string("Failed to construct pipeline state: ") +
        error.description.UTF8String);
  }

  // Allocate memory for input and output matrices
  constexpr auto elem_size = sizeof(float);
  auto buf_A = allocSharedBuffer(dev, M * K * elem_size);
  auto buf_B = allocSharedBuffer(dev, N * K * elem_size);
  auto buf_C = allocSharedBuffer(dev, M * N * elem_size);
  auto queue = [dev newCommandQueue];
  auto do_compute = ^() {
    @autoreleasepool {
      auto cmdBuffer = [queue commandBuffer];
      auto encoder = [cmdBuffer computeCommandEncoder];
      std::vector<unsigned> sizes = {M, K, N, 0};
      [encoder setComputePipelineState:cpl];
      [encoder setBuffer:buf_A offset:0 atIndex:0];
      [encoder setBuffer:buf_B offset:0 atIndex:1];
      [encoder setBuffer:buf_C offset:0 atIndex:2];
      [encoder setBytes:sizes.data()
                 length:sizeof(uint32_t) * sizes.size()
                atIndex:3];
      MTLSize group_size;
      if constexpr (col_div == 1) {
        const auto maxTpG = [cpl maxTotalThreadsPerThreadgroup];
        group_size =
            MTLSizeMake(std::min(static_cast<decltype(M)>(maxTpG), M), 1, 1);
      } else {
        group_size = MTLSizeMake(8, 8, 1);
      }
      [encoder dispatchThreads:MTLSizeMake(N / col_div, M, 1)
          threadsPerThreadgroup:group_size];
      [encoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
    }
  };

  // Capture execution, if MTL_CAPTURE_ENABLED envvar is defined
  auto captureManager = [MTLCaptureManager sharedCaptureManager];
  auto captureDescriptor = [MTLCaptureDescriptor new];
  auto gpuTraceString = [NSString stringWithFormat:@"%s.gputrace", shader_name.c_str()];
  captureDescriptor.captureObject = queue;
  captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  captureDescriptor.outputURL = [NSURL fileURLWithPath:gpuTraceString];
  [captureManager startCaptureWithDescriptor:captureDescriptor error:nil];
  do_compute();
  [captureManager stopCapture];

  // Benchmark performance (including dispatch overhead)
  auto gflops = (M * N * K * 1e-9) / measure_time(200, do_compute);
  std::cout << "Perf of " << shader_name << " dim " << M << "x" << N << "x" << K
            << " is " << gflops << " GFLOPs" << std::endl;
}

int main() {
  unsigned M, N, K;
  std::tie(M, N, K) = std::make_tuple(32, 4128, 4096);
  id<MTLDevice> device = getMetalDevice();
  std::cout << "Using device " << device.name.UTF8String << std::endl;
  benchmark_gemm(device, naive_gemm, M, N, K);
  benchmark_gemm(device, vec4_gemm, M, N, K);
  benchmark_gemm<4>(device, mat4_gemm, M, N, K);
}
