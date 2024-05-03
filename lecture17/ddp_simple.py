# modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import profile

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.w = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        return self.w * 7.0 * x


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    with profile() as prof:
        x = torch.tensor(dist.get_rank(), dtype=torch.float)
        y = ddp_model(x)
        print(f"rank {rank}: y=w*7*x: {y.item()}={ddp_model.module.w.item()}*7*{x.item()}")
        print(f"rank {rank}: dy/dw=7*x: {7.0*x.item()}")
        y.backward()
        print(f"rank {rank}: reduced dy/dw: {ddp_model.module.w.grad.item()}")
    if rank == 0:
        print("exporting trace")
        prof.export_chrome_trace("trace_ddp_simple.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Running")
    demo_basic()

# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 ddp_simple.py
