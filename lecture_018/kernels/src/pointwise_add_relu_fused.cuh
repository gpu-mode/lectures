#include <torch/types.h>

torch::Tensor add_relu_fusion(torch::Tensor in_out, const torch::Tensor& in);