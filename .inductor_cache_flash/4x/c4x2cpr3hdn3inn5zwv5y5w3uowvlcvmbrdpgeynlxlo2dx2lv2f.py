# AOT ID: ['6_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /work/.inductor_cache_flash/tu/ctulbb2my3w6dxr34ezm7e5vb2qvhv4ke6hpbjki6o6g27cwgf6l.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states => clone, convert_element_type, var_mean
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %clone : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg3_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf0,%buf1,%buf2
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1720320, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/tz/ctzpeadkyzxg55bg3xszwxxxlyhe33ohckg6euo3o46prnigq7tn.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states => clone, convert_element_type, var_mean
# Graph fragment:
#   %buf0 : Tensor "f32[1, 1024, 1, 6][6144, 1, 6144, 1024]cuda:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[1, 1024, 1, 6][6144, 1, 6144, 1024]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f32[1, 1024, 1, 6][6144, 1, 6144, 1024]cuda:0" = PlaceHolder[target=buf2]
#   %clone : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg3_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf4
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 90112, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 1024*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 1024*r0_1), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/lp/clpuplyh54qqbcnfdviggzvfngexbyrc7p2kmsyzc4riiq7sjbpc.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, clone, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %getitem_1 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf4 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf4]
#   %arg1_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %clone : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg3_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg2_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_poi_fused_native_layer_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1581056, 'x': 3148800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1, 1], 768.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1, 1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 768*y0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/mm/cmm7hkl3wz74ptmq4bcxwaur3sh3dhdmpykovbg4cjxyosuwuyof.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   hidden_states_1 => add_2
#   hidden_states_2 => clone_1, convert_element_type_14, var_mean_1
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_3 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_14 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf17,%buf18,%buf19
triton_red_fused_add_native_layer_norm_view_3 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'y': 1024, 'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 3, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1572864, 'x': 110592, 'r0_': 1572864}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 6
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    tmp5_mean = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (y0 + 1024*r0_2 + 131072*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 768*y0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 2)
    tmp5 = tmp6[:, :, None]
    tmp9 = tmp7[:, :, None]
    tmp10 = tmp8[:, :, None]
    tl.store(out_ptr0 + (x1 + 6*y0), tmp5, xmask)
    tl.store(out_ptr1 + (x1 + 6*y0), tmp9, xmask)
    tl.store(out_ptr2 + (x1 + 6*y0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/iy/ciyuew5kjt7yywj2k4fczzeljoxxz2b2itzr75qsnd5aztthk2mq.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   hidden_states_1 => add_2
#   hidden_states_2 => clone_1, convert_element_type_14, var_mean_1
# Graph fragment:
#   %buf17 : Tensor "f32[1, 1024, 1, 6][6144, 6, 6144, 1]cuda:0" = PlaceHolder[target=buf17]
#   %buf18 : Tensor "f32[1, 1024, 1, 6][6144, 6, 6144, 1]cuda:0" = PlaceHolder[target=buf18]
#   %buf19 : Tensor "f32[1, 1024, 1, 6][6144, 6, 6144, 1]cuda:0" = PlaceHolder[target=buf19]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_14 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_12,%buf21
triton_per_fused_add_native_layer_norm_view_4 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16384, 'r0_': 55296}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 6
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 6*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/mx/cmxbb2a25pje7kphiuljhpdla2sfacfxvyhglez2qu5hl2dzrlwq.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   hidden_states_1 => add_2
#   hidden_states_2 => add_3, add_4, clone_1, convert_element_type_14, convert_element_type_15, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_3 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_12 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf21 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf21]
#   %arg12_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_14 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %getitem_12), kwargs = {})
#   %add_3 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg12_1), kwargs = {})
#   %add_4 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg13_1), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_15
triton_poi_fused_add_native_layer_norm_view_5 = async_compile.triton('triton_poi_fused_add_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1581056, 'x': 4721664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 768*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1, 1], 768.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 768*y0), tmp19, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/kn/cknqu2q5ti7q4dxfbkjljb5uvhwgfonhliuaqc32tyuxejqt7fde.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => view_14
#   hidden_states_4 => add_5, add_6, convert_element_type_19, convert_element_type_20, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, tanh
# Graph fragment:
#   %addmm_4 : Tensor "bf16[1024, 3072][3072, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %view_14 : Tensor "bf16[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 1024, 3072]), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_14, torch.float32), kwargs = {})
#   %mul_8 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 0.5), kwargs = {})
#   %mul_4 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %convert_element_type_19), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %convert_element_type_19), kwargs = {})
#   %mul_6 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_5 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_7,), kwargs = {})
#   %add_6 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_9 : Tensor "f32[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_6), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 1024, 3072][3145728, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20
triton_poi_fused_gelu_view_6 = async_compile.triton('triton_poi_fused_gelu_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_view_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/cr/ccrqqgx2jc7ahpcllbrze26z653co3ly4hab37dxapxns45cfjne.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   hidden_states_1 => add_2
#   hidden_states_5 => view_16
#   hidden_states_6 => add_7
#   hidden_states_7 => add_8, add_9, clone_2, convert_element_type_24, convert_element_type_25, mul_10, mul_11, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_3 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %getitem_14 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_14]
#   %buf28 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf28]
#   %arg18_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %view_16 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1024, 768]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_16), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_7,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_24 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_2, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_24, %getitem_14), kwargs = {})
#   %add_8 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_13, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_11 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg18_1), kwargs = {})
#   %add_9 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg19_1), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   return %getitem_14,%buf28,%convert_element_type_25
triton_red_fused_add_native_layer_norm_view_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572864, 'r0_': 6294528}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp7
        tmp20 = tl.full([1, 1], 768.0, tl.float32)
        tmp21 = (tmp11 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/ti/ctiascmmngq63gxqhthv373f72bmatvfgf7qdabxibxzgbcflb55.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   attn_output_7 => view_28
#   hidden_states_1 => add_2
#   hidden_states_5 => view_16
#   hidden_states_6 => add_7
#   hidden_states_8 => add_10
#   hidden_states_9 => add_11, add_12, clone_3, convert_element_type_38, convert_element_type_39, mul_12, mul_13, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_3 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %getitem_25 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_25]
#   %buf42 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf42]
#   %arg28_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %view_16 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1024, 768]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 1024, 768]), kwargs = {})
#   %add_10 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_28), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_10,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_38 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_38, %getitem_25), kwargs = {})
#   %add_11 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_12 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_13 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %arg28_1), kwargs = {})
#   %add_12 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg29_1), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   return %getitem_25,%buf42,%convert_element_type_39
triton_red_fused_add_native_layer_norm_view_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572864, 'r0_': 7867392}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp9
        tmp24 = tl.full([1, 1], 768.0, tl.float32)
        tmp25 = (tmp13 / tmp24)
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/f5/cf55n4q3564xiv6g5tmxsptxpbbszs2gsbzlfwszavmgef2jgvyg.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   attn_output_7 => view_28
#   hidden_states_1 => add_2
#   hidden_states_12 => view_32
#   hidden_states_13 => add_15
#   hidden_states_14 => add_16, add_17, clone_4, convert_element_type_48, convert_element_type_49, mul_20, mul_21, rsqrt_4, sub_4, var_mean_4
#   hidden_states_5 => view_16
#   hidden_states_6 => add_7
#   hidden_states_8 => add_10
# Graph fragment:
#   %arg3_1 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_3 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %addmm_11 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_11]
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_15]
#   %getitem_27 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf50 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf50]
#   %arg34_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %view_12 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1024, 768]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_12), kwargs = {})
#   %view_16 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1024, 768]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 1024, 768]), kwargs = {})
#   %add_10 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_28), kwargs = {})
#   %view_32 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_11, [1, 1024, 768]), kwargs = {})
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_32), kwargs = {})
#   %clone_4 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_15,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_48 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_4, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_48, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_48, %getitem_27), kwargs = {})
#   %add_16 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_20 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_21 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg34_1), kwargs = {})
#   %add_17 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg35_1), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   return %add_15,%getitem_27,%buf50,%convert_element_type_49
triton_red_fused_add_native_layer_norm_view_9 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1572864, 'r0_': 12585984}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(r0_mask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask & xmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp11
        tmp20 = tl.full([1, 1], 768.0, tl.float32)
        tmp21 = (tmp15 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/se/cse5y65iileuuj3sq7qnnzysoo7d7x46g7imt66k5rwx6bsmeoxv.py
# Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_44
#   hidden_states_15 => add_18
#   hidden_states_16 => add_19, add_20, clone_5, convert_element_type_62, convert_element_type_63, mul_22, mul_23, rsqrt_5, sub_5, var_mean_5
# Graph fragment:
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_15 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %getitem_38 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_38]
#   %buf64 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf64]
#   %arg44_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %view_44 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 1024, 768]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_44), kwargs = {})
#   %clone_5 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_18,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_62 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_62, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_62, %getitem_38), kwargs = {})
#   %add_19 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_37, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_23 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %arg44_1), kwargs = {})
#   %add_20 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %arg45_1), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.bfloat16), kwargs = {})
#   return %getitem_38,%buf64,%convert_element_type_63
triton_per_fused_add_native_layer_norm_view_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 6294528}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tmp3 - tmp13
    tmp21 = tl.full([1, 1], 768.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tl.full([1, 1], 1e-06, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/zp/czpqw2zjiekhe7h5zx6g2igqtwgp73zcdzu2nvmmvcl2teiemjdh.py
# Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_44
#   hidden_states_15 => add_18
#   hidden_states_19 => view_48
#   hidden_states_20 => add_23
#   hidden_states_21 => add_24, add_25, clone_6, convert_element_type_72, convert_element_type_73, mul_30, mul_31, rsqrt_6, sub_6, var_mean_6
# Graph fragment:
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_15 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %addmm_17 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %getitem_40 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_40]
#   %buf71 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf71]
#   %arg50_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg50_1]
#   %arg51_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %view_44 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 1024, 768]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_44), kwargs = {})
#   %view_48 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, 1024, 768]), kwargs = {})
#   %add_23 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_48), kwargs = {})
#   %clone_6 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_23,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_72 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_6, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_72, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_72, %getitem_40), kwargs = {})
#   %add_24 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_39, 1e-06), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %mul_30 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_31 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %arg50_1), kwargs = {})
#   %add_25 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %arg51_1), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_25, torch.bfloat16), kwargs = {})
#   return %getitem_40,%buf71,%convert_element_type_73
triton_per_fused_add_native_layer_norm_view_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 7867392}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([1, 1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tmp5 - tmp15
    tmp23 = tl.full([1, 1], 768.0, tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tl.full([1, 1], 1e-06, tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/fv/cfvdrrr2lunk2lwrvb7xme4dbbr533upwilmxia5id7h3skuwdnl.py
# Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_44
#   attn_output_15 => view_60
#   hidden_states_15 => add_18
#   hidden_states_19 => view_48
#   hidden_states_20 => add_23
#   hidden_states_22 => add_26
#   hidden_states_23 => add_27, add_28, clone_7, convert_element_type_86, convert_element_type_87, mul_32, mul_33, rsqrt_7, sub_7, var_mean_7
# Graph fragment:
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_15 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %addmm_17 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %addmm_21 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_21]
#   %getitem_51 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_51]
#   %buf85 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf85]
#   %arg60_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %arg61_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %view_44 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 1024, 768]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_44), kwargs = {})
#   %view_48 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, 1024, 768]), kwargs = {})
#   %add_23 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_48), kwargs = {})
#   %view_60 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_21, [1, 1024, 768]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_60), kwargs = {})
#   %clone_7 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_26,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_86 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_7, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_86, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_86, %getitem_51), kwargs = {})
#   %add_27 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-06), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %mul_32 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg60_1), kwargs = {})
#   %add_28 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg61_1), kwargs = {})
#   %convert_element_type_87 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_28, torch.bfloat16), kwargs = {})
#   return %getitem_51,%buf85,%convert_element_type_87
triton_per_fused_add_native_layer_norm_view_12 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 9440256}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 768, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tmp7 - tmp17
    tmp25 = tl.full([1, 1], 768.0, tl.float32)
    tmp26 = (tmp23 / tmp25)
    tmp27 = tl.full([1, 1], 1e-06, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp37, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/cr/ccrnf5a7mdq7aaibd2ka72wcbdbnjhtg7p23swg6cbunozw2m7zn.py
# Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_44
#   attn_output_15 => view_60
#   hidden_states_15 => add_18
#   hidden_states_19 => view_48
#   hidden_states_20 => add_23
#   hidden_states_22 => add_26
#   hidden_states_26 => view_64
#   hidden_states_27 => add_31
#   hidden_states_28 => add_32, add_33, clone_8, convert_element_type_96, convert_element_type_97, mul_40, mul_41, rsqrt_8, sub_8, var_mean_8
# Graph fragment:
#   %add_15 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_15 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %addmm_17 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %addmm_21 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_21]
#   %addmm_23 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_23]
#   %add_31 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_31]
#   %getitem_53 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_53]
#   %buf93 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf93]
#   %arg66_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg66_1]
#   %arg67_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg67_1]
#   %view_44 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 1024, 768]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_44), kwargs = {})
#   %view_48 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, 1024, 768]), kwargs = {})
#   %add_23 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_48), kwargs = {})
#   %view_60 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_21, [1, 1024, 768]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_60), kwargs = {})
#   %view_64 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_23, [1, 1024, 768]), kwargs = {})
#   %add_31 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_64), kwargs = {})
#   %clone_8 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_31,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_96 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_8, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_96, %getitem_53), kwargs = {})
#   %add_32 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_40 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_41 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %arg66_1), kwargs = {})
#   %add_33 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %arg67_1), kwargs = {})
#   %convert_element_type_97 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.bfloat16), kwargs = {})
#   return %add_31,%getitem_53,%buf93,%convert_element_type_97
triton_per_fused_add_native_layer_norm_view_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 14158848}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([1, 1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = tl.full([1, 1], 768.0, tl.float32)
    tmp28 = (tmp25 / tmp27)
    tmp29 = tl.full([1, 1], 1e-06, tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/nn/cnnrzknka3a44672thoucply2ho2hxjszpo3gtdkyogtv6lmsjpo.py
# Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, last_hidden_state], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_43 => view_172
#   attn_output_47 => view_188
#   hidden_states_71 => add_82
#   hidden_states_75 => view_176
#   hidden_states_76 => add_87
#   hidden_states_78 => add_90
#   hidden_states_82 => view_192
#   hidden_states_83 => add_95
#   last_hidden_state => add_96, add_97, clone_24, convert_element_type_288, convert_element_type_289, mul_120, mul_121, rsqrt_24, sub_24, var_mean_24
# Graph fragment:
#   %add_79 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=add_79]
#   %addmm_63 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_63]
#   %addmm_65 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_65]
#   %addmm_69 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_69]
#   %addmm_71 : Tensor "bf16[1024, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_71]
#   %convert_element_type_288 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=convert_element_type_288]
#   %getitem_157 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=getitem_157]
#   %buf265 : Tensor "f32[1, 1024, 1][1024, 1, 1024]cuda:0" = PlaceHolder[target=buf265]
#   %arg194_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg194_1]
#   %arg195_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg195_1]
#   %view_172 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_63, [1, 1024, 768]), kwargs = {})
#   %add_82 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, %view_172), kwargs = {})
#   %view_176 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_65, [1, 1024, 768]), kwargs = {})
#   %add_87 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %view_176), kwargs = {})
#   %view_188 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_69, [1, 1024, 768]), kwargs = {})
#   %add_90 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %view_188), kwargs = {})
#   %view_192 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_71, [1, 1024, 768]), kwargs = {})
#   %add_95 : Tensor "bf16[1, 1024, 768][786432, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %view_192), kwargs = {})
#   %clone_24 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_95,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_288 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_24, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_288, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_288, %getitem_157), kwargs = {})
#   %add_96 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-06), kwargs = {})
#   %rsqrt_24 : Tensor "f32[1, 1024, 1][1024, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_96,), kwargs = {})
#   %mul_120 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_121 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %arg194_1), kwargs = {})
#   %add_97 : Tensor "f32[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %arg195_1), kwargs = {})
#   %convert_element_type_289 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_97, torch.bfloat16), kwargs = {})
#   return %convert_element_type_288,%getitem_157,%buf265,%convert_element_type_289
triton_per_fused_add_native_layer_norm_view_14 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 11013120}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([1, 1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = tl.full([1, 1], 768.0, tl.float32)
    tmp28 = (tmp25 / tmp27)
    tmp29 = tl.full([1, 1], 1e-06, tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')

def partition_0(args):
    arg3_1, arg1_1, arg2_1, arg5_1, arg4_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg15_1, arg14_1, arg17_1, arg16_1, arg18_1, arg19_1, arg21_1, arg20_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg31_1, arg30_1, arg33_1, arg32_1, arg34_1, arg35_1, arg37_1, arg36_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg47_1, arg46_1, arg49_1, arg48_1, arg50_1, arg51_1, arg53_1, arg52_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg63_1, arg62_1, arg65_1, arg64_1, arg66_1, arg67_1, arg69_1, arg68_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg79_1, arg78_1, arg81_1, arg80_1, arg82_1, arg83_1, arg85_1, arg84_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg95_1, arg94_1, arg97_1, arg96_1, arg98_1, arg99_1, arg101_1, arg100_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg111_1, arg110_1, arg113_1, arg112_1, arg114_1, arg115_1, arg117_1, arg116_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg127_1, arg126_1, arg129_1, arg128_1, arg130_1, arg131_1, arg133_1, arg132_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg143_1, arg142_1, arg145_1, arg144_1, arg146_1, arg147_1, arg149_1, arg148_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg159_1, arg158_1, arg161_1, arg160_1, arg162_1, arg163_1, arg165_1, arg164_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg175_1, arg174_1, arg177_1, arg176_1, arg178_1, arg179_1, arg181_1, arg180_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg191_1, arg190_1, arg193_1, arg192_1, arg194_1, arg195_1 = args
    args.clear()
    assert_size_stride(arg3_1, (1, 1024, 768), (786432, 1, 1024))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, ), (1, ))
    assert_size_stride(arg30_1, (3072, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, ), (1, ))
    assert_size_stride(arg78_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (3072, ), (1, ))
    assert_size_stride(arg110_1, (3072, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 3072), (3072, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, ), (1, ))
    assert_size_stride(arg126_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (3072, ), (1, ))
    assert_size_stride(arg158_1, (3072, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 3072), (3072, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, ), (1, ))
    assert_size_stride(arg174_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (3072, ), (1, ))
    assert_size_stride(arg190_1, (3072, 768), (768, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (768, 3072), (3072, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1024, 1, 6), (6144, 1, 6144, 1024), torch.float32)
        buf1 = empty_strided_cuda((1, 1024, 1, 6), (6144, 1, 6144, 1024), torch.float32)
        buf2 = empty_strided_cuda((1, 1024, 1, 6), (6144, 1, 6144, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(arg3_1, buf0, buf1, buf2, 6144, 128, stream=stream0)
        buf3 = empty_strided_cuda((1, 1024, 1), (1024, 1, 1024), torch.float32)
        buf4 = empty_strided_cuda((1, 1024, 1), (1024, 1, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_1.run(buf0, buf1, buf2, buf3, buf4, 1024, 6, stream=stream0)
        buf6 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_2.run(arg3_1, buf3, buf4, arg1_1, arg2_1, buf6, 1024, 768, stream=stream0)
        del arg1_1
        del arg2_1
        buf7 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf6, (1024, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg4_1
        del arg5_1
        buf8 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf6, (1024, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
        del arg6_1
        del arg7_1
        buf9 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf6, (1024, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del arg8_1
        del arg9_1
        del buf6
        # Topologically Sorted Source Nodes: [linear, view_1, queries, linear_1, view_2, keys, linear_2, view_3, values, attn_output], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf10 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf7, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf9, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        buf11 = buf10[0]
        assert_size_stride(buf11, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf11, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf10
        buf16 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, reshape, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf11, (1024, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf16)
        del arg10_1
        del arg11_1
        buf17 = reinterpret_tensor(buf2, (1, 1024, 1, 6), (6144, 6, 6144, 1), 0); del buf2  # reuse
        buf18 = reinterpret_tensor(buf1, (1, 1024, 1, 6), (6144, 6, 6144, 1), 0); del buf1  # reuse
        buf19 = reinterpret_tensor(buf0, (1, 1024, 1, 6), (6144, 6, 6144, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_view_3.run(arg3_1, buf16, buf17, buf18, buf19, 1024, 6, 128, stream=stream0)
        buf20 = buf4; del buf4  # reuse
        buf21 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_4.run(buf17, buf18, buf19, buf20, buf21, 1024, 6, stream=stream0)
        del buf17
        del buf18
        del buf19
        buf23 = reinterpret_tensor(buf11, (1, 1024, 768), (786432, 768, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_view_5.run(arg3_1, buf16, buf20, buf21, arg12_1, arg13_1, buf23, 1024, 768, stream=stream0)
        del arg12_1
        del arg13_1
        del buf20
        del buf21
        buf24 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf23, (1024, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf24)
        del arg14_1
        del arg15_1
        buf25 = reinterpret_tensor(buf24, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf25, 3145728, stream=stream0)
        buf26 = reinterpret_tensor(buf23, (1024, 768), (768, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg17_1, reinterpret_tensor(buf25, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg16_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf26)
        del arg16_1
        del arg17_1
        buf30 = reinterpret_tensor(buf8, (1, 1024, 768), (786432, 768, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_view_7.run(arg3_1, buf16, buf26, arg18_1, arg19_1, buf30, 1024, 768, stream=stream0)
        del arg18_1
        del arg19_1
        buf31 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg21_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
        del arg20_1
        del arg21_1
        buf32 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg23_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
        del arg22_1
        del arg23_1
        buf33 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg25_1, reinterpret_tensor(buf30, (1024, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
        del arg24_1
        del arg25_1
        del buf30
        # Topologically Sorted Source Nodes: [linear_6, view_4, queries_1, linear_7, view_5, keys_1, linear_8, view_6, values_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf34 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf31, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf32, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf33, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf31
        del buf32
        buf35 = buf34[0]
        assert_size_stride(buf35, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf35, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf34
        buf40 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [transpose_7, reshape_1, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf35, (1024, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf40)
        del arg26_1
        del arg27_1
        buf44 = reinterpret_tensor(buf35, (1, 1024, 768), (786432, 768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_view_8.run(arg3_1, buf16, buf26, buf40, arg28_1, arg29_1, buf44, 1024, 768, stream=stream0)
        del arg28_1
        del arg29_1
        buf45 = reinterpret_tensor(buf25, (1024, 3072), (3072, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9, hidden_states_10], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf44, (1024, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf45)
        del arg30_1
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf46, 3145728, stream=stream0)
        buf47 = reinterpret_tensor(buf44, (1024, 768), (768, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, hidden_states_12], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg33_1, reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg32_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf47)
        del arg32_1
        del arg33_1
        buf48 = reinterpret_tensor(buf16, (1, 1024, 768), (786432, 768, 1), 0); del buf16  # reuse
        buf52 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_view_9.run(buf48, arg3_1, buf26, buf40, buf47, arg34_1, arg35_1, buf52, 1024, 768, stream=stream0)
        del arg34_1
        del arg35_1
        del arg3_1
        buf53 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg37_1, reinterpret_tensor(buf52, (1024, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del arg36_1
        del arg37_1
        buf54 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg39_1, reinterpret_tensor(buf52, (1024, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
        del arg38_1
        del arg39_1
        buf55 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg41_1, reinterpret_tensor(buf52, (1024, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf55)
        del arg40_1
        del arg41_1
        del buf52
        # Topologically Sorted Source Nodes: [linear_12, view_7, queries_2, linear_13, view_8, keys_2, linear_14, view_9, values_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf56 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf53, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf54, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf55, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf53
        buf57 = buf56[0]
        assert_size_stride(buf57, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf57, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf56
        buf62 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [transpose_11, reshape_2, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf57, (1024, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf62)
        del arg42_1
        del arg43_1
        buf66 = reinterpret_tensor(buf57, (1, 1024, 768), (786432, 768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_10.run(buf48, buf62, arg44_1, arg45_1, buf66, 1024, 768, stream=stream0)
        del arg44_1
        del arg45_1
        buf67 = reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_16, hidden_states_17], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf66, (1024, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del arg46_1
        del arg47_1
        buf68 = reinterpret_tensor(buf67, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf68, 3145728, stream=stream0)
        buf69 = reinterpret_tensor(buf66, (1024, 768), (768, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18, hidden_states_19], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg49_1, reinterpret_tensor(buf68, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf69)
        del arg48_1
        del arg49_1
        buf73 = reinterpret_tensor(buf54, (1, 1024, 768), (786432, 768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_11.run(buf48, buf62, buf69, arg50_1, arg51_1, buf73, 1024, 768, stream=stream0)
        del arg50_1
        del arg51_1
        buf74 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg53_1, reinterpret_tensor(buf73, (1024, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf74)
        del arg52_1
        del arg53_1
        buf75 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf73, (1024, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf75)
        del arg54_1
        del arg55_1
        buf76 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf73, (1024, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf76)
        del arg56_1
        del arg57_1
        del buf73
        # Topologically Sorted Source Nodes: [linear_18, view_10, queries_3, linear_19, view_11, keys_3, linear_20, view_12, values_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf77 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf74, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf75, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf76, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf74
        del buf75
        buf78 = buf77[0]
        assert_size_stride(buf78, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf78, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf77
        buf83 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [transpose_15, reshape_3, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf78, (1024, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf83)
        del arg58_1
        del arg59_1
        buf87 = reinterpret_tensor(buf78, (1, 1024, 768), (786432, 768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_12.run(buf48, buf62, buf69, buf83, arg60_1, arg61_1, buf87, 1024, 768, stream=stream0)
        del arg60_1
        del arg61_1
        buf88 = reinterpret_tensor(buf68, (1024, 3072), (3072, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23, hidden_states_24], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf87, (1024, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf88)
        del arg62_1
        del arg63_1
        del buf87
        buf89 = reinterpret_tensor(buf88, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf89, 3145728, stream=stream0)
        buf90 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25, hidden_states_26], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg65_1, reinterpret_tensor(buf89, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg64_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf90)
        del arg64_1
        del arg65_1
        buf91 = buf48; del buf48  # reuse
        buf95 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_13.run(buf91, buf62, buf69, buf83, buf90, arg66_1, arg67_1, buf95, 1024, 768, stream=stream0)
        del arg66_1
        del arg67_1
        del buf62
        buf96 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg69_1, reinterpret_tensor(buf95, (1024, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf96)
        del arg68_1
        del arg69_1
        buf97 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf95, (1024, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
        del arg70_1
        del arg71_1
        buf98 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf95, (1024, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf98)
        del arg72_1
        del arg73_1
        del buf95
        # Topologically Sorted Source Nodes: [linear_24, view_13, queries_4, linear_25, view_14, keys_4, linear_26, view_15, values_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf99 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf96, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf97, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf98, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf96
        buf100 = buf99[0]
        assert_size_stride(buf100, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf100, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf99
        buf105 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [transpose_19, reshape_4, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf100, (1024, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
        del arg74_1
        del arg75_1
        buf109 = reinterpret_tensor(buf100, (1, 1024, 768), (786432, 768, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_30], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_10.run(buf91, buf105, arg76_1, arg77_1, buf109, 1024, 768, stream=stream0)
        del arg76_1
        del arg77_1
        buf110 = reinterpret_tensor(buf89, (1024, 3072), (3072, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_30, hidden_states_31], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf110)
        del arg78_1
        del arg79_1
        buf111 = reinterpret_tensor(buf110, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf111, 3145728, stream=stream0)
        buf112 = reinterpret_tensor(buf109, (1024, 768), (768, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32, hidden_states_33], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg81_1, reinterpret_tensor(buf111, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg80_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf112)
        del arg80_1
        del arg81_1
        buf116 = reinterpret_tensor(buf97, (1, 1024, 768), (786432, 768, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_11.run(buf91, buf105, buf112, arg82_1, arg83_1, buf116, 1024, 768, stream=stream0)
        del arg82_1
        del arg83_1
        buf117 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf117)
        del arg84_1
        del arg85_1
        buf118 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg87_1, reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf118)
        del arg86_1
        del arg87_1
        buf119 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg89_1, reinterpret_tensor(buf116, (1024, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
        del arg88_1
        del arg89_1
        del buf116
        # Topologically Sorted Source Nodes: [linear_30, view_16, queries_5, linear_31, view_17, keys_5, linear_32, view_18, values_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf120 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf117, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf118, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf119, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf117
        del buf118
        buf121 = buf120[0]
        assert_size_stride(buf121, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf121, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf120
        buf126 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [transpose_23, reshape_5, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf121, (1024, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
        del arg90_1
        del arg91_1
        buf130 = reinterpret_tensor(buf121, (1, 1024, 768), (786432, 768, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_12.run(buf91, buf105, buf112, buf126, arg92_1, arg93_1, buf130, 1024, 768, stream=stream0)
        del arg92_1
        del arg93_1
        buf131 = reinterpret_tensor(buf111, (1024, 3072), (3072, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_37, hidden_states_38], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf131)
        del arg94_1
        del arg95_1
        del buf130
        buf132 = reinterpret_tensor(buf131, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf132, 3145728, stream=stream0)
        buf133 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39, hidden_states_40], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg97_1, reinterpret_tensor(buf132, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf133)
        del arg96_1
        del arg97_1
        buf134 = buf91; del buf91  # reuse
        buf138 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_13.run(buf134, buf105, buf112, buf126, buf133, arg98_1, arg99_1, buf138, 1024, 768, stream=stream0)
        del arg98_1
        del arg99_1
        del buf105
        buf139 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del arg100_1
        del arg101_1
        buf140 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf140)
        del arg102_1
        del arg103_1
        buf141 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf141)
        del arg104_1
        del arg105_1
        del buf138
        # Topologically Sorted Source Nodes: [linear_36, view_19, queries_6, linear_37, view_20, keys_6, linear_38, view_21, values_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf142 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf139, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf140, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf141, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf139
        buf143 = buf142[0]
        assert_size_stride(buf143, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf143, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf142
        buf148 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [transpose_27, reshape_6, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf143, (1024, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf148)
        del arg106_1
        del arg107_1
        buf152 = reinterpret_tensor(buf143, (1, 1024, 768), (786432, 768, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_44], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_10.run(buf134, buf148, arg108_1, arg109_1, buf152, 1024, 768, stream=stream0)
        del arg108_1
        del arg109_1
        buf153 = reinterpret_tensor(buf132, (1024, 3072), (3072, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del arg110_1
        del arg111_1
        buf154 = reinterpret_tensor(buf153, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf154, 3145728, stream=stream0)
        buf155 = reinterpret_tensor(buf152, (1024, 768), (768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg113_1, reinterpret_tensor(buf154, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg112_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf155)
        del arg112_1
        del arg113_1
        buf159 = reinterpret_tensor(buf140, (1, 1024, 768), (786432, 768, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, hidden_states_49], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_11.run(buf134, buf148, buf155, arg114_1, arg115_1, buf159, 1024, 768, stream=stream0)
        del arg114_1
        del arg115_1
        buf160 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg117_1, reinterpret_tensor(buf159, (1024, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf160)
        del arg116_1
        del arg117_1
        buf161 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg119_1, reinterpret_tensor(buf159, (1024, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf161)
        del arg118_1
        del arg119_1
        buf162 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg121_1, reinterpret_tensor(buf159, (1024, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf162)
        del arg120_1
        del arg121_1
        del buf159
        # Topologically Sorted Source Nodes: [linear_42, view_22, queries_7, linear_43, view_23, keys_7, linear_44, view_24, values_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf163 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf160, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf161, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf162, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf160
        del buf161
        buf164 = buf163[0]
        assert_size_stride(buf164, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf164, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf163
        buf169 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [transpose_31, reshape_7, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf164, (1024, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf169)
        del arg122_1
        del arg123_1
        buf173 = reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_51], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_12.run(buf134, buf148, buf155, buf169, arg124_1, arg125_1, buf173, 1024, 768, stream=stream0)
        del arg124_1
        del arg125_1
        buf174 = reinterpret_tensor(buf154, (1024, 3072), (3072, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_51, hidden_states_52], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf173, (1024, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf174)
        del arg126_1
        del arg127_1
        del buf173
        buf175 = reinterpret_tensor(buf174, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf175, 3145728, stream=stream0)
        buf176 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53, hidden_states_54], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg129_1, reinterpret_tensor(buf175, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg128_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf176)
        del arg128_1
        del arg129_1
        buf177 = buf134; del buf134  # reuse
        buf181 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, hidden_states_56], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_13.run(buf177, buf148, buf155, buf169, buf176, arg130_1, arg131_1, buf181, 1024, 768, stream=stream0)
        del arg130_1
        del arg131_1
        del buf148
        buf182 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg133_1, reinterpret_tensor(buf181, (1024, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf182)
        del arg132_1
        del arg133_1
        buf183 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg135_1, reinterpret_tensor(buf181, (1024, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf183)
        del arg134_1
        del arg135_1
        buf184 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg137_1, reinterpret_tensor(buf181, (1024, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf184)
        del arg136_1
        del arg137_1
        del buf181
        # Topologically Sorted Source Nodes: [linear_48, view_25, queries_8, linear_49, view_26, keys_8, linear_50, view_27, values_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf185 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf182, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf183, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf184, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf182
        buf186 = buf185[0]
        assert_size_stride(buf186, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf186, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf185
        buf191 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [transpose_35, reshape_8, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf186, (1024, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf191)
        del arg138_1
        del arg139_1
        buf195 = reinterpret_tensor(buf186, (1, 1024, 768), (786432, 768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_58], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_10.run(buf177, buf191, arg140_1, arg141_1, buf195, 1024, 768, stream=stream0)
        del arg140_1
        del arg141_1
        buf196 = reinterpret_tensor(buf175, (1024, 3072), (3072, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_58, hidden_states_59], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf195, (1024, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg142_1
        del arg143_1
        buf197 = reinterpret_tensor(buf196, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf197, 3145728, stream=stream0)
        buf198 = reinterpret_tensor(buf195, (1024, 768), (768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg145_1, reinterpret_tensor(buf197, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf198)
        del arg144_1
        del arg145_1
        buf202 = reinterpret_tensor(buf183, (1, 1024, 768), (786432, 768, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, hidden_states_63], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_11.run(buf177, buf191, buf198, arg146_1, arg147_1, buf202, 1024, 768, stream=stream0)
        del arg146_1
        del arg147_1
        buf203 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg149_1, reinterpret_tensor(buf202, (1024, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf203)
        del arg148_1
        del arg149_1
        buf204 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg151_1, reinterpret_tensor(buf202, (1024, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf204)
        del arg150_1
        del arg151_1
        buf205 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg153_1, reinterpret_tensor(buf202, (1024, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
        del arg152_1
        del arg153_1
        del buf202
        # Topologically Sorted Source Nodes: [linear_54, view_28, queries_9, linear_55, view_29, keys_9, linear_56, view_30, values_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf206 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf203, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf204, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf205, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf203
        del buf204
        buf207 = buf206[0]
        assert_size_stride(buf207, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf207, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf206
        buf212 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [transpose_39, reshape_9, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf207, (1024, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf212)
        del arg154_1
        del arg155_1
        buf216 = reinterpret_tensor(buf207, (1, 1024, 768), (786432, 768, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_65], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_12.run(buf177, buf191, buf198, buf212, arg156_1, arg157_1, buf216, 1024, 768, stream=stream0)
        del arg156_1
        del arg157_1
        buf217 = reinterpret_tensor(buf197, (1024, 3072), (3072, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_65, hidden_states_66], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf216, (1024, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del arg158_1
        del arg159_1
        del buf216
        buf218 = reinterpret_tensor(buf217, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf218, 3145728, stream=stream0)
        buf219 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, hidden_states_68], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg161_1, reinterpret_tensor(buf218, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg160_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf219)
        del arg160_1
        del arg161_1
        buf220 = buf177; del buf177  # reuse
        buf224 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_13.run(buf220, buf191, buf198, buf212, buf219, arg162_1, arg163_1, buf224, 1024, 768, stream=stream0)
        del arg162_1
        del arg163_1
        del buf191
        buf225 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg165_1, reinterpret_tensor(buf224, (1024, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf225)
        del arg164_1
        del arg165_1
        buf226 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg167_1, reinterpret_tensor(buf224, (1024, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del arg166_1
        del arg167_1
        buf227 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg169_1, reinterpret_tensor(buf224, (1024, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del arg168_1
        del arg169_1
        del buf224
        # Topologically Sorted Source Nodes: [linear_60, view_31, queries_10, linear_61, view_32, keys_10, linear_62, view_33, values_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf228 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf225, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf226, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf227, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf225
        buf229 = buf228[0]
        assert_size_stride(buf229, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf229, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf228
        buf234 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [transpose_43, reshape_10, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf229, (1024, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf234)
        del arg170_1
        del arg171_1
        buf238 = reinterpret_tensor(buf229, (1, 1024, 768), (786432, 768, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_72], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_10.run(buf220, buf234, arg172_1, arg173_1, buf238, 1024, 768, stream=stream0)
        del arg172_1
        del arg173_1
        buf239 = reinterpret_tensor(buf218, (1024, 3072), (3072, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf238, (1024, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf239)
        del arg174_1
        del arg175_1
        buf240 = reinterpret_tensor(buf239, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf240, 3145728, stream=stream0)
        buf241 = reinterpret_tensor(buf238, (1024, 768), (768, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74, hidden_states_75], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg177_1, reinterpret_tensor(buf240, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg176_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf241)
        del arg176_1
        del arg177_1
        buf245 = reinterpret_tensor(buf226, (1, 1024, 768), (786432, 768, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_11.run(buf220, buf234, buf241, arg178_1, arg179_1, buf245, 1024, 768, stream=stream0)
        del arg178_1
        del arg179_1
        buf246 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg181_1, reinterpret_tensor(buf245, (1024, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf246)
        del arg180_1
        del arg181_1
        buf247 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg183_1, reinterpret_tensor(buf245, (1024, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf247)
        del arg182_1
        del arg183_1
        buf248 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg185_1, reinterpret_tensor(buf245, (1024, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf248)
        del arg184_1
        del arg185_1
        del buf245
        # Topologically Sorted Source Nodes: [linear_66, view_34, queries_11, linear_67, view_35, keys_11, linear_68, view_36, values_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf249 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf246, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf247, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), reinterpret_tensor(buf248, (1, 12, 1024, 64), (786432, 64, 768, 1), 0), scale=0.125)
        del buf246
        del buf247
        buf250 = buf249[0]
        assert_size_stride(buf250, (1, 12, 1024, 64), (786432, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf250, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf249
        buf255 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [transpose_47, reshape_11, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf250, (1024, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf255)
        del arg186_1
        del arg187_1
        buf259 = reinterpret_tensor(buf250, (1, 1024, 768), (786432, 768, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_79], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_12.run(buf220, buf234, buf241, buf255, arg188_1, arg189_1, buf259, 1024, 768, stream=stream0)
        del arg188_1
        del arg189_1
        buf260 = reinterpret_tensor(buf240, (1024, 3072), (3072, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.addmm(arg191_1, reinterpret_tensor(buf259, (1024, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf260)
        del arg190_1
        del arg191_1
        del buf259
        buf261 = reinterpret_tensor(buf260, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_view_6.run(buf261, 3145728, stream=stream0)
        buf262 = empty_strided_cuda((1024, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, hidden_states_82], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg193_1, reinterpret_tensor(buf261, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg192_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf262)
        del arg192_1
        del arg193_1
        del buf261
        buf267 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, last_hidden_state], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_14.run(buf220, buf234, buf241, buf255, buf262, arg194_1, arg195_1, buf267, 1024, 768, stream=stream0)
        del arg194_1
        del arg195_1
        del buf220
        del buf234
        del buf241
        del buf255
        del buf262
    return (buf267, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1 = args
        args.clear()
        partition0_args = [arg3_1, arg1_1, arg2_1, arg5_1, arg4_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg15_1, arg14_1, arg17_1, arg16_1, arg18_1, arg19_1, arg21_1, arg20_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg31_1, arg30_1, arg33_1, arg32_1, arg34_1, arg35_1, arg37_1, arg36_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg47_1, arg46_1, arg49_1, arg48_1, arg50_1, arg51_1, arg53_1, arg52_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg63_1, arg62_1, arg65_1, arg64_1, arg66_1, arg67_1, arg69_1, arg68_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg79_1, arg78_1, arg81_1, arg80_1, arg82_1, arg83_1, arg85_1, arg84_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg95_1, arg94_1, arg97_1, arg96_1, arg98_1, arg99_1, arg101_1, arg100_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg111_1, arg110_1, arg113_1, arg112_1, arg114_1, arg115_1, arg117_1, arg116_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg127_1, arg126_1, arg129_1, arg128_1, arg130_1, arg131_1, arg133_1, arg132_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg143_1, arg142_1, arg145_1, arg144_1, arg146_1, arg147_1, arg149_1, arg148_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg159_1, arg158_1, arg161_1, arg160_1, arg162_1, arg163_1, arg165_1, arg164_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg175_1, arg174_1, arg177_1, arg176_1, arg178_1, arg179_1, arg181_1, arg180_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg191_1, arg190_1, arg193_1, arg192_1, arg194_1, arg195_1]
        del arg3_1, arg1_1, arg2_1, arg5_1, arg4_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg15_1, arg14_1, arg17_1, arg16_1, arg18_1, arg19_1, arg21_1, arg20_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg31_1, arg30_1, arg33_1, arg32_1, arg34_1, arg35_1, arg37_1, arg36_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg47_1, arg46_1, arg49_1, arg48_1, arg50_1, arg51_1, arg53_1, arg52_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg63_1, arg62_1, arg65_1, arg64_1, arg66_1, arg67_1, arg69_1, arg68_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg79_1, arg78_1, arg81_1, arg80_1, arg82_1, arg83_1, arg85_1, arg84_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg95_1, arg94_1, arg97_1, arg96_1, arg98_1, arg99_1, arg101_1, arg100_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg111_1, arg110_1, arg113_1, arg112_1, arg114_1, arg115_1, arg117_1, arg116_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg127_1, arg126_1, arg129_1, arg128_1, arg130_1, arg131_1, arg133_1, arg132_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg143_1, arg142_1, arg145_1, arg144_1, arg146_1, arg147_1, arg149_1, arg148_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg159_1, arg158_1, arg161_1, arg160_1, arg162_1, arg163_1, arg165_1, arg164_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg175_1, arg174_1, arg177_1, arg176_1, arg178_1, arg179_1, arg181_1, arg180_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg191_1, arg190_1, arg193_1, arg192_1, arg194_1, arg195_1
        (buf267,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf267, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.bool)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 1024, 768), (786432, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
