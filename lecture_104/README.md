# Gluon

- Speakers: Peter Bell, Mario Lezcano, Keren Zhou
- [Slides](./gluon.pdf)

## Notes

Gluon is a lower-level, Triton-like language for tile-based GPU programming that gives kernel experts direct control over layouts, warp specialization, shared memory, and hardware intrinsics. The talk positions it as a deliberate trade: less portability and more manual work in exchange for Blackwell-friendly scheduling and near-SOL performance.

### Why Gluon

- Triton works well for high-level block programming, but Blackwell needs warp-specialized schedules that do not map cleanly onto a block-level model.
- Gluon keeps a familiar Python API while exposing the details the compiler cannot always choose optimally.
- The compiler still handles advanced lowerings, static shared-memory allocation, implicit barriers, and async-proxy fences.

### Core Programming Model

- Tensor layouts are first-class and describe how data is distributed across registers, lanes, warps, and shared memory.
- `gl.warp_specialize` uses a fork-join model with a default partition and worker partitions so different warp groups can run dedicated sub-kernels.
- Shared memory is a compile-time static allocator, which makes communication and software pipelining explicit but still manageable.
- Gluon exposes hardware intrinsics such as tensor cores, TMA, and `mbarrier`, so asynchronous execution can be expressed directly.
- Aggregates package code and data together as immutable objects to make low-level state management safer.

### Linear Layouts

- Gluon generalizes older blocked, slice, shared-memory, and tensor-memory layout classes into linear layouts.
- A layout can be viewed as a function from hardware threads to tensor coordinates.
- Formally, a linear layout is a linear map over `F2`, so layout reasoning becomes binary linear algebra using XOR/AND in place of add/multiply.
- This gives Gluon a single, well-defined layout model and fixes a large class of ad hoc layout bugs from earlier systems.
- Because the compiler knows the algebraic structure of layouts, it can lower register-to-shared-memory moves generically, choose vectorized stores, derive swizzles, and generate optimal shuffle sequences.
- The talk points to the paper [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2](https://arxiv.org/pdf/2505.23819).

### Blackwell Matmul Case Study

- The matmul pipeline stages data through global memory, shared memory, tensor memory, registers, and back out to global memory.
- `tcgen05.mma` uses tensor memory as a staging area for operands and accumulators.
- Two-CTA mode lets neighboring CTAs share B tiles, reducing shared-memory pressure while enabling a larger MMA.
- TMA multicast lets one CTA load a tile once and replicate it across multiple CTAs, cutting L2 traffic.
- Cluster Launch Control enables dynamic work distribution for persistent kernels when the static CTA-to-SM assignment would leave hardware idle.
- The presented B200 results show Gluon matmul performing on par with cuBLAS without exhaustive autotuning.

### Tooling

- Proton supports coarse and fine-grained profiling through standard profiling, instrumentation, and PC sampling backends.
- Sanitizers include `consan` for races inside a Gluon program, `iisan` for invalid TMA alignment, `fpsan` for floating-point-sensitive transformations, and `gsan` for global-memory race detection across CTAs and kernels.
- The linear layout visualizer is available at <https://deep-learning-profiling-tools.github.io/linear-layout-viz/> with code at <https://github.com/Deep-Learning-Profiling-Tools/linear-layout-viz>.

## References

- [Gluon tutorials](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html)
- [Gluon examples](https://triton-lang.org/main/getting-started/examples/gluon/index.html)
