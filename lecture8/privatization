__global__ void histogram_update_naive(int *data, int *hist, int data_size, int hist_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        int data_value = data[idx];
        if (data_value < hist_size) {
            atomicAdd(&hist[data_value], 1); // Potential contention point
        }
    }
}


__global__ void histogram_update_privatized(int *data, int *hist, int data_size, int hist_size) {
    extern __shared__ int private_hist[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;

    // Initialize private histogram in shared memory
    for (int i = thread_id; i < hist_size; i += blockDim.x) {
        private_hist[i] = 0;
    }
    __syncthreads(); // Ensure all threads have initialized their private histograms

    // Update private histogram
    if (idx < data_size) {
        int data_value = data[idx];
        if (data_value < hist_size) {
            atomicAdd(&private_hist[data_value], 1);
        }
    }
    __syncthreads(); // Ensure all threads have finished updating the private histogram

    // Reduce private histograms into global histogram
    for (int i = thread_id; i < hist_size; i += blockDim.x) {
        atomicAdd(&hist[i], private_hist[i]);
    }
}
