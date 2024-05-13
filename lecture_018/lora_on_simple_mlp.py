from functools import partial

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils import parametrize


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float, device: str):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev).to(device)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)).to(device)
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, device):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, device=device
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
assign_lora = partial(LinearWithLoRA, rank=5, alpha=0.5, device=device)

if __name__ == "__main__":
    model = MLP(16).to(device)
    logger.info(model.layers[0])
    for num_layers in range(len(model.layers)):
        if isinstance(model.layers[num_layers], nn.Linear):
            model.layers[num_layers] = assign_lora(model.layers[num_layers])
            model.layers[num_layers].linear.requires_grad = False
    logger.info(model.layers[0])
    logger.info(model)

    logger.info(model(torch.randn(7, 16).to(device)))
    torch.compile(model, fullgraph=True, mode="max-autotune")(torch.randn(7, 16).to(device))