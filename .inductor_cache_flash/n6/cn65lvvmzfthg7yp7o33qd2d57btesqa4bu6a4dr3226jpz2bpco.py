# AOT ID: ['4_inference']
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



# kernel path: /work/.inductor_cache_flash/mp/cmpradlolomiqyhkocsy2xwe4dhynftgu2donci6u2ebzl3yrv26.py
# Topologically Sorted Source Nodes: [boundaries], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   boundaries => add, convert_element_type, iota, mul
# Graph fragment:
#   %iota : Tensor "i64[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (31,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : Tensor "f32[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul : Tensor "f32[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.03125), kwargs = {})
#   %add : Tensor "f32[31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.03125), kwargs = {})
#   return %add
triton_poi_fused_arange_0 = async_compile.triton('triton_poi_fused_arange_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 248}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.03125, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 + tmp2
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/rw/crws6wmxuexfchooj4dt2vkergsrhmu2foxucjokmz4lnmpk7f7g.py
# Topologically Sorted Source Nodes: [h_indices, getitem_2, getitem, nb_patches_h, step_h, getitem_3, fractional_coords_h, fractional_coords_h_1, fractional_coords_h_2, boundaries, bucket_coords_h, w_indices, getitem_4, getitem_1, nb_patches_w, step_w, getitem_5, fractional_coords_w, fractional_coords_w_1, fractional_coords_w_2, bucket_coords_w], Original ATen: [aten.arange, aten.unsqueeze, aten.select, aten.sum, aten.reciprocal, aten.mul, aten.clamp, aten._to_copy, aten.bucketize]
# Source node to ATen node mapping:
#   boundaries => add, convert_element_type, iota, mul
#   bucket_coords_h => bucketize
#   bucket_coords_w => bucketize_1
#   fractional_coords_h => mul_5
#   fractional_coords_h_1 => clamp_max
#   fractional_coords_h_2 => convert_element_type_3
#   fractional_coords_w => mul_6
#   fractional_coords_w_1 => clamp_max_1
#   fractional_coords_w_2 => convert_element_type_4
#   getitem => select
#   getitem_1 => select_1
#   getitem_2 => unsqueeze
#   getitem_3 => unsqueeze_1
#   getitem_4 => unsqueeze_2
#   getitem_5 => unsqueeze_3
#   h_indices => add_1, convert_element_type_1, iota_1, mul_3
#   nb_patches_h => sum_1
#   nb_patches_w => sum_2
#   step_h => mul_1, reciprocal
#   step_w => mul_2, reciprocal_1
#   w_indices => add_2, convert_element_type_2, iota_2, mul_4
# Graph fragment:
#   %arg3_1 : Tensor "b8[1, 32, 32][1024, 32, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sum_1 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=sum_1]
#   %add : Tensor "f32[31][1]cuda:0" = PlaceHolder[target=add]
#   %sum_2 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=sum_2]
#   %iota_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_3 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_1 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 0), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 0), kwargs = {})
#   %select : Tensor "b8[1, 32][1024, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 2, 0), kwargs = {})
#   %sum_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select, [1]), kwargs = {})
#   %reciprocal : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sum_1,), kwargs = {})
#   %mul_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_1, 1), kwargs = {})
#   %mul_5 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %clamp_max : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_5, 0.999999), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %iota : Tensor "i64[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (31,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : Tensor "f32[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul : Tensor "f32[31][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.03125), kwargs = {})
#   %add : Tensor "f32[31][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.03125), kwargs = {})
#   %bucketize : Tensor "i64[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bucketize.Tensor](args = (%convert_element_type_3, %add), kwargs = {right: True})
#   %iota_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_4 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_2 : Tensor "i64[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 0), kwargs = {})
#   %select_1 : Tensor "b8[1, 32][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg3_1, 1, 0), kwargs = {})
#   %sum_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_1, [1]), kwargs = {})
#   %reciprocal_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sum_2,), kwargs = {})
#   %mul_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1.0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_2, 1), kwargs = {})
#   %mul_6 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %clamp_max_1 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_6, 0.999999), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.bfloat16), kwargs = {})
#   %bucketize_1 : Tensor "i64[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bucketize.Tensor](args = (%convert_element_type_4, %add), kwargs = {right: True})
#   return %sum_1,%sum_2,%bucketize,%bucketize_1
triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1 = async_compile.triton('triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr2': '*i64', 'out_ptr3': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': {AutotuneHint.ONE_ELEMENT_PER_THREAD}, 'kernel_name': 'triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 1056}}
)
@triton.jit
def triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (32*r0_0), None, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr0 + (r0_0), None).to(tl.int1)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None].to(tl.int64)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.int64)
    tmp10 = tmp4.to(tl.float32)
    tmp11 = tl.full([1, 1], 1, tl.int32)
    tmp12 = (tmp11 / tmp10)
    tmp13 = tl.full([1, 1], 1.0, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = r0_0
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tl.full([1, 1], 0.999999, tl.float32)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = triton_helpers.bucketize_binary_search(tmp21, in_ptr1, 31, 31, 1, 0, tl.int64, True, None, None, None, )
    tmp23 = tmp9.to(tl.float32)
    tmp24 = (tmp11 / tmp23)
    tmp25 = tmp24 * tmp13
    tmp26 = tmp16 * tmp25
    tmp27 = triton_helpers.minimum(tmp26, tmp18)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = triton_helpers.bucketize_binary_search(tmp29, in_ptr1, 31, 31, 1, 0, tl.int64, True, None, None, None, )
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp22, None)
    tl.store(out_ptr3 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp30, None)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/xs/cxsfgfjxvfoh35qp4envepxpnvcghdacjwrhdqq6mbswj4ynt6zv.py
# Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, pos_ids], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   getitem_6 => unsqueeze_4
#   getitem_7 => unsqueeze_5
#   mul_2 => mul_7
#   pos_ids => add_3
# Graph fragment:
#   %bucketize : Tensor "i64[1, 32][32, 1]cuda:0" = PlaceHolder[target=bucketize]
#   %bucketize_1 : Tensor "i64[1, 32][32, 1]cuda:0" = PlaceHolder[target=bucketize_1]
#   %unsqueeze_4 : Tensor "i64[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%bucketize, 2), kwargs = {})
#   %mul_7 : Tensor "i64[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_4, 32), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%bucketize_1, 1), kwargs = {})
#   %add_3 : Tensor "i64[1, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_5), kwargs = {})
#   return %add_3
triton_poi_fused_add_mul_unsqueeze_2 = async_compile.triton('triton_poi_fused_add_mul_unsqueeze_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 256, 'x': 16640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_unsqueeze_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 32, tl.int64)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x1 + 32*y0), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/7m/c7mn2ut2fmdzbxsrc7gd5v6xfowces3s66c5byfxvjm727tyejoi.py
# Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   patch_embeds => convolution
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "bf16[1, 768, 32, 32][786432, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [16, 16], [0], [1, 1], False, [0], 1), kwargs = {})
#   return %buf6
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 262144}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1179648, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 262144*y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/4w/c4w5hhgdw47csds3celh4bqgt7huqmibxtowbw5vetfiuxgzcjwn.py
# Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   patch_embeds => convolution
# Graph fragment:
#   %arg1_1 : Tensor "bf16[768, 3, 16, 16][768, 256, 16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "bf16[1, 768, 32, 32][786432, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [16, 16], [0], [1, 1], False, [0], 1), kwargs = {})
#   return %buf7
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/is/cis3iwhcgifoyup4itfovvuaarcsfni7tmg2otwqukxeydmlc5w4.py
# Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   patch_embeds => convolution
# Graph fragment:
#   %buf8 : Tensor "bf16[1, 768, 32, 32][786432, 1, 24576, 768]cuda:0" = PlaceHolder[target=buf8]
#   %arg2_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "bf16[1, 768, 32, 32][786432, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [16, 16], [0], [1, 1], False, [0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1574400, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 768*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 1024*y0), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/2u/c2uz3lkczjyafxpjgw2wz5xtortmud7ri6vesqv6c3knt3vqq6xp.py
# Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.full]
# Source node to ATen node mapping:
#   position_ids => full_default
# Graph fragment:
#   %full_default : Tensor "i64[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1024], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_full_6 = async_compile.triton('triton_poi_fused_full_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')

def partition_0(args):
    arg3_1, arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg3_1, (1, 32, 32), (1024, 32, 1))
    assert_size_stride(arg0_1, (1, 3, 512, 512), (786432, 262144, 512, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((31, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [boundaries], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_0.run(buf1, 31, stream=stream0)
        buf2 = empty_strided_cuda((1, 32), (32, 1), torch.int64)
        buf4 = empty_strided_cuda((1, 32), (32, 1), torch.int64)
        # Topologically Sorted Source Nodes: [h_indices, getitem_2, getitem, nb_patches_h, step_h, getitem_3, fractional_coords_h, fractional_coords_h_1, fractional_coords_h_2, boundaries, bucket_coords_h, w_indices, getitem_4, getitem_1, nb_patches_w, step_w, getitem_5, fractional_coords_w, fractional_coords_w_1, fractional_coords_w_2, bucket_coords_w], Original ATen: [aten.arange, aten.unsqueeze, aten.select, aten.sum, aten.reciprocal, aten.mul, aten.clamp, aten._to_copy, aten.bucketize]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_arange_bucketize_clamp_mul_reciprocal_select_sum_unsqueeze_1.run(arg3_1, buf1, buf2, buf4, 1, 32, stream=stream0)
        del buf1
        buf5 = empty_strided_cuda((1, 32, 32), (1024, 32, 1), torch.int64)
        # Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, pos_ids], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_unsqueeze_2.run(buf2, buf4, buf5, 32, 32, stream=stream0)
        del buf2
        del buf4
        buf6 = empty_strided_cuda((1, 3, 512, 512), (786432, 1, 1536, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg0_1, buf6, 3, 262144, stream=stream0)
        del arg0_1
        buf7 = empty_strided_cuda((768, 3, 16, 16), (768, 1, 48, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(arg1_1, buf7, 2304, 256, stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(16, 16), padding=(0,), dilation=(1, 1), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf8, (1, 768, 32, 32), (786432, 1, 24576, 768), 'torch.ops.aten.convolution.default')
        del buf7
        buf9 = reinterpret_tensor(buf6, (1, 768, 32, 32), (786432, 1024, 32, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [patch_embeds], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_5.run(buf8, arg2_1, buf9, 768, 1024, stream=stream0)
        del arg2_1
        del buf8
        buf10 = empty_strided_cuda((1, 1024), (1024, 1), torch.int64)
        # Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_6.run(buf10, 1024, stream=stream0)
    return (buf5, buf9, buf10, arg3_1, )


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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        partition0_args = [arg3_1, arg0_1, arg1_1, arg2_1]
        del arg0_1, arg1_1, arg2_1
        (buf5, buf9, buf10, arg3_1) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf5, (1, 1024), (1024, 1), 0), reinterpret_tensor(arg3_1, (1, 1024), (1024, 1), 0), reinterpret_tensor(buf9, (1, 1024, 768), (786432, 1, 1024), 0), buf10, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 3, 512, 512), (786432, 262144, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.bool)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
