#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_add_kernel_9969bdda_0123(void);
void load_add_kernel_9969bdda_0123(void);
// tt-linker: add_kernel_9969bdda_0123:CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements:64_warps1xstages3
CUresult add_kernel_9969bdda_0123(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
