__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b; }


std::tuple<torch::Tensor, torch::Tensor> your_function_name(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int n = Q.size(0);
    int n_inp = K.size(0);
    int d = Q.size(1);
    
    assert (d == V.size(1) && "Size mismatch!");
    assert (d == K.size(1) && "Size mismatch!");
    assert (K.size(0) == V.size(0) && "Size mismatch!");
    auto out = torch::zeros({n, d}, Q.options());
    auto out_l = torch::zeros({n,}, Q.options());

    float scaling = 1.0f / sqrt((float)d);

    int T_r = cdiv(n, B_r);
    int T_c = cdiv(n_inp, B_c);

    dim3 blocks(T_r, 1);      
    dim3 tpb(block_dim_x, block_dim_y); 
    your_function_name_k<<<blocks, tpb>>>(
        out.data_ptr<float>(),
        out_l.data_ptr<float>(),
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(), 
        scaling,
        n,
        T_r,
        T_c
    );
    return std::make_tuple(out, out_l);
}