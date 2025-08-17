constexpr int B_r = 16;
constexpr int B_c = 16;
constexpr int d = 128;
constexpr int block_dim_x = 16;
constexpr int block_dim_y = 16;
constexpr int d_over_dim_x = d / block_dim_x;
constexpr int B_r_over_dim_y = B_r / block_dim_y;


extern "C" __global__ void flash_attention_k(
    float *out, 
    float *out_l, 
    float *Q,
    float *K, 
    float *V, 
    float scaling, 
    int n, 
    int T_r, 
    int T_c
) {
    // Thread indices
    int tid_x = threadIdx.x; 
    int tid_y = threadIdx.y; 
    int dim_y = blockDim.y; 
    int dim_x = blockDim.x; 

    // Shared memory buffers for Q, K, V blocks
    __shared__ float Q_i[B_r][d];       // 16 x 128
    __shared__ float K_j[B_c][d];       // 16 x 128
    __shared__ float V_j[B_c][d];       // 16 x 128
    __shared__ float S[B_r][B_c];     //16 X 16

    // Local accumulators per thread for output block
    float l_i[B_r_over_dim_y];
    float m_i[B_r_over_dim_y];
    float O_i[B_r_over_dim_y][d_over_dim_x];

    // Loop over output tile blocks (T_r)
    for (int i = 0; i < T_r; i++) {
        // Init O_i, l_i, m_i into registers
        // Load Q_i tile into shared mem
        for (int ii = tid_y; ii < B_r; ii += dim_y) {
            for (int dd = tid_x; dd < d; dd += dim_x) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
            }
        }
        __syncthreads();
        for (int ii = 0; ii < B_r_over_dim_y; ii ++) {
            for (int dd = 0; dd < d_over_dim_x; dd ++) {
                O_i[ii][dd] = 0.f;
            }
            l_i[ii] = 0.f;
            m_i[ii] = 1e-30f;
        }
        __syncthreads();

        for (int j = 0; j < T_c; j++){
            // Load K_j, V_j into shared memory
            for (int jj = tid_y; jj < B_c; jj += dim_y) {
                for (int dd = tid_x; dd < d; dd += dim_x) {
                    K_j[jj][dd] = K[(jj + j * B_c) * d + dd];
                    V_j[jj][dd] = V[(jj + j * B_c) * d + dd];
                }
            }
            __syncthreads();
            // S[ii][jj] = scaling * Q_i @ K_j.T
            for (int ii = tid_x; ii < B_r; ii += dim_x) {
                for (int jj = tid_y; jj < B_c; jj += dim_y) {
                    float S_ij = 0.0f;
                    for (int dd = 0; dd < d; dd ++) {
                        S_ij += Q_i[ii][dd] * K_j[jj][dd];
                    }
                    S_ij = scaling * S_ij;
                    S[ii][jj] = S_ij;
                }
            }
            __syncthreads();
            for (int ii = 0; ii < B_r_over_dim_y; ii ++) {
                float m = m_i[ii];
                float last_m = m;
                for (int jj = 0; jj < B_c; jj++) {
                    if (m < S[ii * dim_y + tid_y][jj]) {
                        m = S[ii * dim_y + tid_y][jj];
                    }
                }
                m_i[ii] = m;
                float l = expf(last_m - m) * l_i[ii];

                for (int dd = 0; dd < d_over_dim_x; dd ++) {
                    O_i[ii][dd] *= expf(last_m - m);
                }
                for (int jj = 0; jj < B_c; jj++) {
                    float P_ij = expf(S[ii * dim_y + tid_y][jj] - m);
                    l += P_ij;
                    for (int dd = 0; dd < d_over_dim_x; dd ++) {
                        O_i[ii][dd] +=  P_ij * V_j[jj][dd * dim_x + tid_x];
                    }
                }
                l_i[ii] = l;
            }
        }
        __syncthreads();
        for (int ii = 0; ii < B_r_over_dim_y; ii ++) {
            for (int dd = 0; dd < d_over_dim_x; dd ++) {
                out[(ii * dim_y + tid_y + i * B_r) * d + dd * dim_x + tid_x] = O_i[ii][dd] / l_i[ii];
            }
            out_l[ii * dim_y + tid_y + i * B_r] = m_i[ii] + logf(l_i[ii]);
        }
        __syncthreads();
    }
}



#ifdef STANDALONE_TEST
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { 
    return (a + b - 1) / b; 
}

void print_matrix(const char* name, float* mat, int rows, int cols, int max_print = 4) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < std::min(rows, max_print); i++) {
        for (int j = 0; j < std::min(cols, max_print); j++) {
            printf("%.4f ", mat[i * cols + j]);
        }
        if (cols > max_print) printf("...");
        printf("\n");
    }
    if (rows > max_print) printf("...\n");
    printf("\n");
}

int main() {
    // Test dimensions
    const int n = 32;      // output sequence length
    const int n_inp = 32;  // input sequence length  
    const int d = 128;     // feature dimension
    
    printf("Testing Flash Attention with n=%d, n_inp=%d, d=%d\n", n, n_inp, d);
    
    // Allocate host memory
    float *h_Q = new float[n * d];
    float *h_K = new float[n_inp * d];  
    float *h_V = new float[n_inp * d];
    float *h_out = new float[n * d];
    float *h_out_l = new float[n];
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < n * d; i++) {
        h_Q[i] = dist(gen);
    }
    for (int i = 0; i < n_inp * d; i++) {
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }
    
    // Print some input data
    print_matrix("Q", h_Q, n, d);
    print_matrix("K", h_K, n_inp, d);
    print_matrix("V", h_V, n_inp, d);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_out, *d_out_l;
    cudaMalloc(&d_Q, n * d * sizeof(float));
    cudaMalloc(&d_K, n_inp * d * sizeof(float));
    cudaMalloc(&d_V, n_inp * d * sizeof(float));
    cudaMalloc(&d_out, n * d * sizeof(float));
    cudaMalloc(&d_out_l, n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_Q, h_Q, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, n_inp * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, n_inp * d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up kernel parameters
    float scaling = 1.0f / sqrtf((float)d);
    int T_r = cdiv(n, B_r);
    int T_c = cdiv(n_inp, B_c);
    
    printf("Kernel params: scaling=%.6f, T_r=%d, T_c=%d\n", scaling, T_r, T_c);
    printf("Block sizes: B_r=%d, B_c=%d, block_dim=(%d,%d)\n", 
           B_r, B_c, block_dim_x, block_dim_y);
    
    // Launch kernel
    dim3 blocks(1, 1);
    dim3 threads(block_dim_x, block_dim_y);
    
    printf("Launching kernel...\n");
    flash_attention_k<<<blocks, threads>>>(
        d_out, d_out_l, d_Q, d_K, d_V, 
        scaling, n, n_inp, T_r, T_c
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("Kernel completed successfully!\n");
    
    // Copy results back
    cudaMemcpy(h_out, d_out, n * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_l, d_out_l, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    print_matrix("Output O", h_out, n, d);
    
    printf("Output L (log-sum-exp):\n");
    for (int i = 0; i < std::min(n, 8); i++) {
        printf("L[%d] = %.4f\n", i, h_out_l[i]);
    }
    printf("\n");
    
    // Basic sanity checks
    bool has_nan = false, has_inf = false;
    for (int i = 0; i < n * d; i++) {
        if (isnan(h_out[i])) has_nan = true;
        if (isinf(h_out[i])) has_inf = true;
    }
    for (int i = 0; i < n; i++) {
        if (isnan(h_out_l[i])) has_nan = true;
        if (isinf(h_out_l[i])) has_inf = true;
    }
    
    if (has_nan) printf("WARNING: Output contains NaN values!\n");
    if (has_inf) printf("WARNING: Output contains Inf values!\n");
    if (!has_nan && !has_inf) printf("SUCCESS: No NaN or Inf values detected.\n");
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K; 
    delete[] h_V;
    delete[] h_out;
    delete[] h_out_l;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    cudaFree(d_out_l);
    
    return 0;
}
#endif
