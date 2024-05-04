# modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile
import torch.optim as optim

SIZE = 4000


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(SIZE, SIZE)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(SIZE, SIZE)
        self.net3 = nn.Linear(SIZE, SIZE)

    def forward(self, x):
        return self.net3(self.relu(self.net2(self.relu(self.net1(x)))))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    model = ToyModel().to(rank)
    ddp_model = DDP(model, bucket_cap_mb=25, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    with profile(
        record_shapes=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for i in range(10):
            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(1000, SIZE, device=rank))
            labels = torch.randn(1000, SIZE, device=rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()
    if rank == 0:
        prof.export_chrome_trace("trace_ddp_example.json")


if __name__ == "__main__":
    demo_basic()

# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 ddp_example.py
