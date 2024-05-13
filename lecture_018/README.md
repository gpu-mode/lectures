# Fused Kernels

## Abstract

With focus on performance to get the most out of hardware, fusing of kernels has been a popular technique. At times, researchers/practitioners will re-write their code in native cuda or cpu kernels to get optimal performance, but projects such as torch.compile aim to make this simpler. Talk will focus on generating fused kernels and how to leverage torch.compile to be able to do that. We will shift a bit from all LLM talk and look into recommendation algorithms. In the process, we will work on creating fused kernels (triton and cuda) with the help of `torch.compile`. 

## Code and other artifacts

- Lecture Data: https://github.com/kapilsh/cuda-mode-lecture/tree/main/data
- How to open chrome trace: chrome://tracing
- DLRM Blog Post: https://ai.meta.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/
- DLRM Paper: https://arxiv.org/pdf/1906.00091
- DLRM github repo: https://github.com/facebookresearch/dlrm
- Criteo Dataset: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
- [Pytorch Profiler with Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?source=post_page-----2cb7e0fef30e--------------------------------)
- [TORCH_LOGS with torch.compile](https://pytorch.org/tutorials/recipes/torch_logs.html#beta-using-torch-logs-python-api-with-torch-compile)
- LoRA Paper: https://arxiv.org/abs/2106.09685
- LoRA from scratch: https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
- Netron: https://netron.app/
- GPUs go brrr https://horace.io/brrr_intro.html
