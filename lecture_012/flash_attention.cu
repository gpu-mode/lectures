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

