# https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59/2
import torch
from torch.profiler import profile, record_function, ProfilerActivity
def main():
    for _ in range(100):
        a = torch.square(torch.randn(10000, 10000).cuda())

if __name__ == "__main__":
    main()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")