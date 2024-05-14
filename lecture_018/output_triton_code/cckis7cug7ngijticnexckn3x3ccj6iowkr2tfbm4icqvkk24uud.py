
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_ksharma/y3/cy3fsrxo27bzpyxwxq3ojr6hj2pmndinlgkcigaibcnhcybwf6fq.py
# Source Nodes: [l__self___dense_layer_mlp_fc_layers_1], Original ATen: [aten.relu]
# l__self___dense_layer_mlp_fc_layers_1 => relu
triton_poi_fused_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ksharma/2b/c2btikpopq5qf4s6xew4nt44gueumsrl2xgy3mhn6ekexl5czu4h.py
# Source Nodes: [l__self___dense_layer_mlp_fc_layers_3], Original ATen: [aten.relu]
# l__self___dense_layer_mlp_fc_layers_3 => relu_1
triton_poi_fused_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_ksharma/sg/csgbajlo3ygdkyr7wdentvmufckjkxft3a22vsdj5bzx4rhsme3g.py
# Source Nodes: [l__self___dense_layer_mlp_fc_layers_5], Original ATen: [aten.relu]
# l__self___dense_layer_mlp_fc_layers_5 => relu_2
triton_poi_fused_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_ksharma/r5/cr5ule4qvsqlsurei6qpaqj3uoofpqwghhc5xono7nlio65csa2h.py
# Source Nodes: [cat_1], Original ATen: [aten.cat]
# cat_1 => cat
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_ksharma/24/c24fqapvxytsobfe4ufawgers6trhmxhycnc3mocaloyr757o2c6.py
# Source Nodes: [embeddings], Original ATen: [aten.embedding]
# embeddings => embedding
triton_poi_fused_embedding_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (26*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 1234907
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 1234907), "index out of bounds: 0 <= tmp10 < 1234907")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/bn/cbn3u7uxfhlhcqmlsx6o76lqmkucydxr7qla4usqwmn7szkjm6nv.py
# Source Nodes: [embeddings_1], Original ATen: [aten.embedding]
# embeddings_1 => embedding_1
triton_poi_fused_embedding_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (1 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 19682
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 19682), "index out of bounds: 0 <= tmp10 < 19682")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/ce/ccefj2jwj7fgffu62mfw3wjt53ax32eebedwyyedzq4rpch6eo57.py
# Source Nodes: [embeddings_2], Original ATen: [aten.embedding]
# embeddings_2 => embedding_2
triton_poi_fused_embedding_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (2 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (2))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 13779
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 13779), "index out of bounds: 0 <= tmp10 < 13779")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/rb/crb7hlxod4hvawkydy24sbc2i6qfahf37yfce3eq6ck6bkjtqyei.py
# Source Nodes: [embeddings_3], Original ATen: [aten.embedding]
# embeddings_3 => embedding_3
triton_poi_fused_embedding_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (3 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (3))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 6866
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 6866), "index out of bounds: 0 <= tmp10 < 6866")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/i3/ci3qmnvwqmvzt4uecw5aukqd4wdzsgsrzefbr2asxelpftbhgudz.py
# Source Nodes: [embeddings_4], Original ATen: [aten.embedding]
# embeddings_4 => embedding_4
triton_poi_fused_embedding_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (4 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 18489
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 18489), "index out of bounds: 0 <= tmp10 < 18489")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/kz/ckzizzmvo4odjvhlqv66rzu2b5l3b5wkq5gqztv74y7bpra7ijml.py
# Source Nodes: [embeddings_5], Original ATen: [aten.embedding]
# embeddings_5 => embedding_5
triton_poi_fused_embedding_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (5 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (5))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 3
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 3), "index out of bounds: 0 <= tmp10 < 3")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/kb/ckbltn7bhsbcf6hp3vqmsatzbvhvwvjrvdqiytftfxuipdammjqr.py
# Source Nodes: [embeddings_6], Original ATen: [aten.embedding]
# embeddings_6 => embedding_6
triton_poi_fused_embedding_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (6 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (6))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 6263
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 6263), "index out of bounds: 0 <= tmp10 < 6263")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/tf/ctf67gmwgi575woeo6an7266tbnl6vvzo2vl44pafxsok3bq6gzz.py
# Source Nodes: [embeddings_7], Original ATen: [aten.embedding]
# embeddings_7 => embedding_7
triton_poi_fused_embedding_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (7 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (7))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 1234
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 1234), "index out of bounds: 0 <= tmp10 < 1234")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/rq/crqveac4eal6vzatvu56dodwni32nelgxlixgkwjx6din22gwar5.py
# Source Nodes: [embeddings_8], Original ATen: [aten.embedding]
# embeddings_8 => embedding_8
triton_poi_fused_embedding_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (8 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (8))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 49
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 49), "index out of bounds: 0 <= tmp10 < 49")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/z7/cz7n6lixabwcktedfo27iuyejgpou3ja4tsba726piicsqze4sog.py
# Source Nodes: [embeddings_9], Original ATen: [aten.embedding]
# embeddings_9 => embedding_9
triton_poi_fused_embedding_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (9 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (9))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 854680
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 854680), "index out of bounds: 0 <= tmp10 < 854680")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/ro/cronqtajfyoirv3eu4bgptj6roxonmpysqrcrgezrqskxt4zwrgy.py
# Source Nodes: [embeddings_10], Original ATen: [aten.embedding]
# embeddings_10 => embedding_10
triton_poi_fused_embedding_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (10 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (10))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 114026
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 114026), "index out of bounds: 0 <= tmp10 < 114026")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/qm/cqm3yddfbxc4vhkysse32kqumygdo5zodreoqm35chj2f6honosy.py
# Source Nodes: [embeddings_11], Original ATen: [aten.embedding]
# embeddings_11 => embedding_11
triton_poi_fused_embedding_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (11 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (11))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 75735
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 75735), "index out of bounds: 0 <= tmp10 < 75735")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/wh/cwhg2znezzxqu6zzsuuwzja7dyg6ywa3p7xqep3d22bmbketnowf.py
# Source Nodes: [embeddings_12], Original ATen: [aten.embedding]
# embeddings_12 => embedding_12
triton_poi_fused_embedding_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (12 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (12))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 10
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 10), "index out of bounds: 0 <= tmp10 < 10")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/vs/cvshbt2tknmqmyq6k5dc4crjrhxmbboygoypf4hijqwutztrxg5q.py
# Source Nodes: [embeddings_13], Original ATen: [aten.embedding]
# embeddings_13 => embedding_13
triton_poi_fused_embedding_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (13 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (13))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 2159
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 2159), "index out of bounds: 0 <= tmp10 < 2159")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/jh/cjh5k3kewqi2xfxvydqurs3odj7foyjaj2fcwoaofthekshl6s45.py
# Source Nodes: [embeddings_14], Original ATen: [aten.embedding]
# embeddings_14 => embedding_14
triton_poi_fused_embedding_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (14 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (14))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 7532
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 7532), "index out of bounds: 0 <= tmp10 < 7532")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/37/c3763xtu65le2t5np6pjt7uj6vdo6z2kjhqjuqn3wxgoozycszjw.py
# Source Nodes: [embeddings_15], Original ATen: [aten.embedding]
# embeddings_15 => embedding_15
triton_poi_fused_embedding_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (15 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (15))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 61
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 61), "index out of bounds: 0 <= tmp10 < 61")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/py/cpy4a7kriz3jphwkuxggzexvcfyabhixtdwnlphaq2hdf5fmppak.py
# Source Nodes: [embeddings_16], Original ATen: [aten.embedding]
# embeddings_16 => embedding_16
triton_poi_fused_embedding_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (16 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (16))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 4
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 4), "index out of bounds: 0 <= tmp10 < 4")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/gg/cggmzwapsna5ygcsycfne2lgzsonsbb2q6i7sxrykvnbdbfzkrok.py
# Source Nodes: [embeddings_17], Original ATen: [aten.embedding]
# embeddings_17 => embedding_17
triton_poi_fused_embedding_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (17 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (17))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 918
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 918), "index out of bounds: 0 <= tmp10 < 918")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/od/codujg3sg4vy3idd3w5mqmftcctkvaqwsju25pfnxwc2bglryvgh.py
# Source Nodes: [embeddings_18], Original ATen: [aten.embedding]
# embeddings_18 => embedding_18
triton_poi_fused_embedding_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (18 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (18))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 14
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 14), "index out of bounds: 0 <= tmp10 < 14")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/be/cbe46cfas6k5rwxtvq3p2kk5mi37sslr3lh3zhrrmhg4hdsedvtv.py
# Source Nodes: [embeddings_19], Original ATen: [aten.embedding]
# embeddings_19 => embedding_19
triton_poi_fused_embedding_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (19 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (19))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 1307783
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 1307783), "index out of bounds: 0 <= tmp10 < 1307783")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/f2/cf2bvl6tbqghkhr5zzwp4dmluphvv54muzhkvvop2jiernbqshaf.py
# Source Nodes: [embeddings_20], Original ATen: [aten.embedding]
# embeddings_20 => embedding_20
triton_poi_fused_embedding_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (20 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (20))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 404742
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 404742), "index out of bounds: 0 <= tmp10 < 404742")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/4s/c4shff5ni2qymsczudkvztwgtqqthfah2nm4vtw7r5zksj55xzkb.py
# Source Nodes: [embeddings_21], Original ATen: [aten.embedding]
# embeddings_21 => embedding_21
triton_poi_fused_embedding_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (21 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (21))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 1105613
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 1105613), "index out of bounds: 0 <= tmp10 < 1105613")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/45/c4563uxo3tb7fgxkakher6zqwluwl5agxozmw3cghdgdg4ypnane.py
# Source Nodes: [embeddings_22], Original ATen: [aten.embedding]
# embeddings_22 => embedding_22
triton_poi_fused_embedding_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (22 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (22))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 87714
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 87714), "index out of bounds: 0 <= tmp10 < 87714")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/vp/cvpljojitbdc2ftok32hzqxqbaql6h6lzrszq37vgl7kx77327hr.py
# Source Nodes: [embeddings_23], Original ATen: [aten.embedding]
# embeddings_23 => embedding_23
triton_poi_fused_embedding_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (23 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (23))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 9031
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 9031), "index out of bounds: 0 <= tmp10 < 9031")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/me/cme4i7xbp7ueo6hlygvvzlfdi4ukh3eybhjxokm43ejxvkvmm5mg.py
# Source Nodes: [embeddings_24], Original ATen: [aten.embedding]
# embeddings_24 => embedding_24
triton_poi_fused_embedding_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (24 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (24))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 76
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 76), "index out of bounds: 0 <= tmp10 < 76")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/af/cafjpi4wag7q3nkwl7woqnirlevshsbam6quvyygy4zp33qqfux6.py
# Source Nodes: [embeddings_25], Original ATen: [aten.embedding]
# embeddings_25 => embedding_25
triton_poi_fused_embedding_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (25 + (26*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (25))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 % tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tl.where(((tmp5 != 0) & ((tmp5 < 0) != (tmp4 < 0))), tmp6, tmp5)
    tmp8 = tmp7 + 33
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 33), "index out of bounds: 0 <= tmp10 < 33")
    tmp11 = tl.load(in_ptr2 + (x0 + (16*tmp10)), None)
    tl.store(out_ptr0 + (x0 + (432*x1)), tmp11, None)
''')


# kernel path: /tmp/torchinductor_ksharma/ab/cabkfn2fbpklz6otc6q5hsgcf34zf6zdhizijshufnfepspq4tmy.py
# Source Nodes: [result], Original ATen: [aten.sigmoid, aten.squeeze]
# result => sigmoid
triton_poi_fused_sigmoid_squeeze_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_squeeze_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 13), (13, 1))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (256, 512), (512, 1))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (64, 256), (256, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (16, 64), (64, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (1234907, 16), (16, 1))
    assert_size_stride(arg9_1, (19682, 16), (16, 1))
    assert_size_stride(arg10_1, (13779, 16), (16, 1))
    assert_size_stride(arg11_1, (6866, 16), (16, 1))
    assert_size_stride(arg12_1, (18489, 16), (16, 1))
    assert_size_stride(arg13_1, (3, 16), (16, 1))
    assert_size_stride(arg14_1, (6263, 16), (16, 1))
    assert_size_stride(arg15_1, (1234, 16), (16, 1))
    assert_size_stride(arg16_1, (49, 16), (16, 1))
    assert_size_stride(arg17_1, (854680, 16), (16, 1))
    assert_size_stride(arg18_1, (114026, 16), (16, 1))
    assert_size_stride(arg19_1, (75735, 16), (16, 1))
    assert_size_stride(arg20_1, (10, 16), (16, 1))
    assert_size_stride(arg21_1, (2159, 16), (16, 1))
    assert_size_stride(arg22_1, (7532, 16), (16, 1))
    assert_size_stride(arg23_1, (61, 16), (16, 1))
    assert_size_stride(arg24_1, (4, 16), (16, 1))
    assert_size_stride(arg25_1, (918, 16), (16, 1))
    assert_size_stride(arg26_1, (14, 16), (16, 1))
    assert_size_stride(arg27_1, (1307783, 16), (16, 1))
    assert_size_stride(arg28_1, (404742, 16), (16, 1))
    assert_size_stride(arg29_1, (1105613, 16), (16, 1))
    assert_size_stride(arg30_1, (87714, 16), (16, 1))
    assert_size_stride(arg31_1, (9031, 16), (16, 1))
    assert_size_stride(arg32_1, (76, 16), (16, 1))
    assert_size_stride(arg33_1, (33, 16), (16, 1))
    assert_size_stride(arg34_1, (512, 186624), (186624, 1))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (256, 512), (512, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (1, 256), (256, 1))
    assert_size_stride(arg39_1, (1, ), (1, ))
    assert_size_stride(arg40_1, (26, ), (1, ))
    assert_size_stride(arg41_1, (128, 13), (13, 1))
    assert_size_stride(arg42_1, (128, 26), (26, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(arg41_1, reinterpret_tensor(arg0_1, (13, 512), (1, 13), 0), out=buf0)
        del arg0_1
        del arg41_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__self___dense_layer_mlp_fc_layers_1], Original ATen: [aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_relu_0.run(buf1, arg1_1, 65536, grid=grid(65536), stream=stream0)
        del arg1_1
        buf2 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___dense_layer_mlp_fc_layers_1], Original ATen: [aten.relu]
        extern_kernels.mm(buf1, reinterpret_tensor(arg2_1, (512, 256), (1, 512), 0), out=buf2)
        del arg2_1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__self___dense_layer_mlp_fc_layers_3], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf3, arg3_1, 32768, grid=grid(32768), stream=stream0)
        del arg3_1
        buf4 = empty((128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___dense_layer_mlp_fc_layers_3], Original ATen: [aten.relu]
        extern_kernels.mm(buf3, reinterpret_tensor(arg4_1, (256, 64), (1, 256), 0), out=buf4)
        del arg4_1
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [l__self___dense_layer_mlp_fc_layers_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_2.run(buf5, arg5_1, 8192, grid=grid(8192), stream=stream0)
        del arg5_1
        buf6 = empty((128, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [dense_out, l__self___dense_layer_mlp_fc_layers_5], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.addmm(reinterpret_tensor(arg7_1, (128, 16), (0, 1), 0), buf5, reinterpret_tensor(arg6_1, (64, 16), (1, 64), 0), alpha=1, beta=1, out=buf6)
        del arg6_1
        del arg7_1
        del buf5
        buf34 = empty((128, 432), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf34, (128, 16), (432, 1), 0)  # alias
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(buf6, buf7, 2048, grid=grid(2048), stream=stream0)
        del buf6
        buf8 = reinterpret_tensor(buf34, (128, 16), (432, 1), 16)  # alias
        # Source Nodes: [embeddings], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_4.run(arg42_1, arg40_1, arg8_1, buf8, 2048, grid=grid(2048), stream=stream0)
        del arg8_1
        buf9 = reinterpret_tensor(buf34, (128, 16), (432, 1), 32)  # alias
        # Source Nodes: [embeddings_1], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_5.run(arg42_1, arg40_1, arg9_1, buf9, 2048, grid=grid(2048), stream=stream0)
        del arg9_1
        buf10 = reinterpret_tensor(buf34, (128, 16), (432, 1), 48)  # alias
        # Source Nodes: [embeddings_2], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_6.run(arg42_1, arg40_1, arg10_1, buf10, 2048, grid=grid(2048), stream=stream0)
        del arg10_1
        buf11 = reinterpret_tensor(buf34, (128, 16), (432, 1), 64)  # alias
        # Source Nodes: [embeddings_3], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_7.run(arg42_1, arg40_1, arg11_1, buf11, 2048, grid=grid(2048), stream=stream0)
        del arg11_1
        buf12 = reinterpret_tensor(buf34, (128, 16), (432, 1), 80)  # alias
        # Source Nodes: [embeddings_4], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_8.run(arg42_1, arg40_1, arg12_1, buf12, 2048, grid=grid(2048), stream=stream0)
        del arg12_1
        buf13 = reinterpret_tensor(buf34, (128, 16), (432, 1), 96)  # alias
        # Source Nodes: [embeddings_5], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_9.run(arg42_1, arg40_1, arg13_1, buf13, 2048, grid=grid(2048), stream=stream0)
        del arg13_1
        buf14 = reinterpret_tensor(buf34, (128, 16), (432, 1), 112)  # alias
        # Source Nodes: [embeddings_6], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_10.run(arg42_1, arg40_1, arg14_1, buf14, 2048, grid=grid(2048), stream=stream0)
        del arg14_1
        buf15 = reinterpret_tensor(buf34, (128, 16), (432, 1), 128)  # alias
        # Source Nodes: [embeddings_7], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_11.run(arg42_1, arg40_1, arg15_1, buf15, 2048, grid=grid(2048), stream=stream0)
        del arg15_1
        buf16 = reinterpret_tensor(buf34, (128, 16), (432, 1), 144)  # alias
        # Source Nodes: [embeddings_8], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_12.run(arg42_1, arg40_1, arg16_1, buf16, 2048, grid=grid(2048), stream=stream0)
        del arg16_1
        buf17 = reinterpret_tensor(buf34, (128, 16), (432, 1), 160)  # alias
        # Source Nodes: [embeddings_9], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_13.run(arg42_1, arg40_1, arg17_1, buf17, 2048, grid=grid(2048), stream=stream0)
        del arg17_1
        buf18 = reinterpret_tensor(buf34, (128, 16), (432, 1), 176)  # alias
        # Source Nodes: [embeddings_10], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_14.run(arg42_1, arg40_1, arg18_1, buf18, 2048, grid=grid(2048), stream=stream0)
        del arg18_1
        buf19 = reinterpret_tensor(buf34, (128, 16), (432, 1), 192)  # alias
        # Source Nodes: [embeddings_11], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_15.run(arg42_1, arg40_1, arg19_1, buf19, 2048, grid=grid(2048), stream=stream0)
        del arg19_1
        buf20 = reinterpret_tensor(buf34, (128, 16), (432, 1), 208)  # alias
        # Source Nodes: [embeddings_12], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_16.run(arg42_1, arg40_1, arg20_1, buf20, 2048, grid=grid(2048), stream=stream0)
        del arg20_1
        buf21 = reinterpret_tensor(buf34, (128, 16), (432, 1), 224)  # alias
        # Source Nodes: [embeddings_13], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_17.run(arg42_1, arg40_1, arg21_1, buf21, 2048, grid=grid(2048), stream=stream0)
        del arg21_1
        buf22 = reinterpret_tensor(buf34, (128, 16), (432, 1), 240)  # alias
        # Source Nodes: [embeddings_14], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_18.run(arg42_1, arg40_1, arg22_1, buf22, 2048, grid=grid(2048), stream=stream0)
        del arg22_1
        buf23 = reinterpret_tensor(buf34, (128, 16), (432, 1), 256)  # alias
        # Source Nodes: [embeddings_15], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_19.run(arg42_1, arg40_1, arg23_1, buf23, 2048, grid=grid(2048), stream=stream0)
        del arg23_1
        buf24 = reinterpret_tensor(buf34, (128, 16), (432, 1), 272)  # alias
        # Source Nodes: [embeddings_16], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_20.run(arg42_1, arg40_1, arg24_1, buf24, 2048, grid=grid(2048), stream=stream0)
        del arg24_1
        buf25 = reinterpret_tensor(buf34, (128, 16), (432, 1), 288)  # alias
        # Source Nodes: [embeddings_17], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_21.run(arg42_1, arg40_1, arg25_1, buf25, 2048, grid=grid(2048), stream=stream0)
        del arg25_1
        buf26 = reinterpret_tensor(buf34, (128, 16), (432, 1), 304)  # alias
        # Source Nodes: [embeddings_18], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_22.run(arg42_1, arg40_1, arg26_1, buf26, 2048, grid=grid(2048), stream=stream0)
        del arg26_1
        buf27 = reinterpret_tensor(buf34, (128, 16), (432, 1), 320)  # alias
        # Source Nodes: [embeddings_19], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_23.run(arg42_1, arg40_1, arg27_1, buf27, 2048, grid=grid(2048), stream=stream0)
        del arg27_1
        buf28 = reinterpret_tensor(buf34, (128, 16), (432, 1), 336)  # alias
        # Source Nodes: [embeddings_20], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_24.run(arg42_1, arg40_1, arg28_1, buf28, 2048, grid=grid(2048), stream=stream0)
        del arg28_1
        buf29 = reinterpret_tensor(buf34, (128, 16), (432, 1), 352)  # alias
        # Source Nodes: [embeddings_21], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_25.run(arg42_1, arg40_1, arg29_1, buf29, 2048, grid=grid(2048), stream=stream0)
        del arg29_1
        buf30 = reinterpret_tensor(buf34, (128, 16), (432, 1), 368)  # alias
        # Source Nodes: [embeddings_22], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_26.run(arg42_1, arg40_1, arg30_1, buf30, 2048, grid=grid(2048), stream=stream0)
        del arg30_1
        buf31 = reinterpret_tensor(buf34, (128, 16), (432, 1), 384)  # alias
        # Source Nodes: [embeddings_23], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_27.run(arg42_1, arg40_1, arg31_1, buf31, 2048, grid=grid(2048), stream=stream0)
        del arg31_1
        buf32 = reinterpret_tensor(buf34, (128, 16), (432, 1), 400)  # alias
        # Source Nodes: [embeddings_24], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_28.run(arg42_1, arg40_1, arg32_1, buf32, 2048, grid=grid(2048), stream=stream0)
        del arg32_1
        buf33 = reinterpret_tensor(buf34, (128, 16), (432, 1), 416)  # alias
        # Source Nodes: [embeddings_25], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_29.run(arg42_1, arg40_1, arg33_1, buf33, 2048, grid=grid(2048), stream=stream0)
        del arg33_1
        del arg40_1
        del arg42_1
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        del buf18
        del buf19
        del buf20
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        del buf7
        del buf8
        del buf9
        buf35 = empty((128, 432, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (128, 432, 1), (432, 1, 1), 0), reinterpret_tensor(buf34, (128, 1, 432), (432, 1, 1), 0), out=buf35)
        del buf34
        buf36 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 186624), (186624, 1), 0), reinterpret_tensor(arg34_1, (186624, 512), (1, 186624), 0), out=buf36)
        del arg34_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [l__self___prediction_layer_mlp_fc_layers_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf37, arg35_1, 65536, grid=grid(65536), stream=stream0)
        del arg35_1
        buf38 = buf3; del buf3  # reuse
        # Source Nodes: [l__self___prediction_layer_mlp_fc_layers_1], Original ATen: [aten.relu]
        extern_kernels.mm(buf37, reinterpret_tensor(arg36_1, (512, 256), (1, 512), 0), out=buf38)
        del arg36_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [l__self___prediction_layer_mlp_fc_layers_3], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf39, arg37_1, 32768, grid=grid(32768), stream=stream0)
        del arg37_1
        buf40 = empty((128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___prediction_layer_mlp_fc_layers_3], Original ATen: [aten.relu]
        extern_kernels.mm(buf39, reinterpret_tensor(arg38_1, (256, 1), (1, 256), 0), out=buf40)
        del arg38_1
        del buf39
        buf41 = reinterpret_tensor(buf40, (128, ), (1, ), 0); del buf40  # reuse
        # Source Nodes: [result], Original ATen: [aten.sigmoid, aten.squeeze]
        triton_poi_fused_sigmoid_squeeze_30.run(buf41, arg39_1, 128, grid=grid(128), stream=stream0)
        del arg39_1
        return (buf41, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 13), (13, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1234907, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((19682, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((13779, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((6866, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((18489, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((6263, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1234, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((49, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((854680, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((114026, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((75735, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((10, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2159, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((7532, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((61, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((918, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((14, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1307783, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((404742, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1105613, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((87714, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((9031, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((76, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((33, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 186624), (186624, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg41_1 = rand_strided((128, 13), (13, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, 26), (26, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
