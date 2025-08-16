constexpr int B_r = 32;
constexpr int B_c = 16;
constexpr int d = 128;
constexpr int block_dim_x = 16;
constexpr int block_dim_y = 32;
constexpr int o_per_thread_x = d / block_dim_x;
constexpr int o_per_thread_y = B_r / block_dim_y;

#define NEG_INFINITY __int_as_float(0xff800000)


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
    int tid_x = threadIdx.x; // 0..3 (block_x_dim)
    int tid_y = threadIdx.y; // 0..31 (block_y_dim)

    // Shared memory buffers for Q, K, V blocks
    __shared__ float Q_i[B_r][d];       // 16 x 128
    __shared__ float K_j[B_c][d];       // 16 x 128
    __shared__ float V_j[B_c][d];       // 16 x 128
    __shared__ float S[B_r][B_c];     //16 X 16

    // Local accumulators per thread for output block
    float l_i[o_per_thread_y];
    float m_i[o_per_thread_y];
    float O_i[o_per_thread_y][o_per_thread_x];

    // Loop over output tile blocks (T_r)
    for (int i = 0; i < T_r; i++) {
        // Init O_i, l_i, m_i into registers
        // Load Q_i tile into shared mem
        for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
            for (int dd = tid_x; dd < d; dd += blockDim.x) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
                O_i[ii/block_dim_y][dd/block_dim_x] = 0;
            }
            l_i[ii/block_dim_y] = 0.f;
            m_i[ii/block_dim_y] = NEG_INFINITY;
        }
        __syncthreads();

        for (int j = 0; j < T_c; j++){
            // Load K_j, V_j into shared memory
            for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    K_j[jj][dd] = K[(jj + j * B_c) * d + dd];
                    V_j[jj][dd] = V[(jj + j * B_c) * d + dd];
                }
            }
            __syncthreads();
            // S[ii][jj] = scaling * Q_i @ K_j.T
            for (int ii = tid_x; ii < B_r; ii += blockDim.x) {
                for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                    float S_ij = 0.0f;
                    for (int dd = 0; dd < d; dd ++) {
                        S_ij += Q_i[ii][dd] * K_j[jj][dd];
                    }
                    S_ij = scaling * S_ij;
                    S[ii][jj] = S_ij;
                }
            }
            __syncthreads();
            for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
                float m = m_i[ii/block_dim_y];
                float last_m = m;
                for (int jj = 0; jj < B_c; jj++) {
                    if (m < S[ii][jj]) {
                        m = S[ii][jj];
                    }
                }
                m_i[ii/block_dim_y] = m;
                float l = exp(last_m - m) * l_i[ii/block_dim_y];

                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    O_i[ii/block_dim_y][dd/block_dim_x] *= exp(last_m - m);
                }

                for (int jj = 0; jj < B_c; jj++) {
                    float P_ij = exp(S[ii][jj] - m);
                    l += P_ij;
                    for (int dd = tid_x; dd < d; dd += blockDim.x) {
                        O_i[ii/block_dim_y][dd/block_dim_x] +=  P_ij * V_j[jj][dd];
                    }
                }
                l_i[ii/block_dim_y] = l;
            }
        }
        __syncthreads();
        for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
            for (int dd = tid_x; dd < d; dd += blockDim.x) {
                out[(ii + i * B_r) * d + dd] = O_i[ii/block_dim_y][dd/block_dim_x] / l_i[ii/block_dim_y];
            }
            out_l[ii + i * B_r] = m_i[ii/block_dim_y] + log(l_i[ii/block_dim_y]);
        }
    }
}

