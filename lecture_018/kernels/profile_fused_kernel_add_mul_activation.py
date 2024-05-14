import torch
import triton

from triton_fused_add_mul_activation import get_inputs, fused_add_mul_activation_torch, \
    add_mul_activation_torch_scripted

if __name__ == '__main__':
    sample_inputs = [get_inputs(batch_size=65336, weight_size=512, add_manual_seed=False) for _ in range(10)]
    BLOCK_SIZE = 1024
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        # on_trace_ready=partial(trace_handler,
        #                        results_dir="./profiler_logs"),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("kernel_benchmarks",
                                                                worker_name="triton_vs_torchscript"),
        profile_memory=True,
        with_stack=True
    )
    prof.start()
    for sample_input in sample_inputs:
        in_out_tensor, in_tensor, bias = sample_input
        fused_add_mul_activation_torch(in_out_tensor, bias, in_tensor)
        # add_mul_activation_torch_scripted(in_out_tensor, bias, in_tensor)
        prof.step()
    prof.stop()
