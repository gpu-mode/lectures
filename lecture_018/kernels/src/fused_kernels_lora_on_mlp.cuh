#include <torch/types.h>

torch::Tensor fused_add_mul_relu(torch::Tensor in_out,
                                 const torch::Tensor &bias,
                                 const torch::Tensor &in,
                                 const double multiplier);