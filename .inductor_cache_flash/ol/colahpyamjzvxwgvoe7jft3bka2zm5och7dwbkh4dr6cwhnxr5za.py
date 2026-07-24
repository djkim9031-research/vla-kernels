# AOT ID: ['0_inference']
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



# kernel path: /work/.inductor_cache_flash/gv/cgv5p7z6opvr4rsot6xcfl4seglhwdzsknn2nj6q3cx54kxbktbg.py
# Topologically Sorted Source Nodes: [resized_img, mul, img], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view, aten._unsafe_index]
# Source node to ATen node mapping:
#   img => sub_7
#   mul => mul_5
#   resized_img => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add, add_1, add_2, add_3, add_4, add_5, add_6, clamp_max, clamp_max_1, clamp_max_2, clamp_max_3, clamp_min, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota, iota_1, mul, mul_1, mul_2, mul_3, mul_4, sub, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, view
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mul_3 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = PlaceHolder[target=mul_3]
#   %sub_6 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = PlaceHolder[target=sub_6]
#   %add_6 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = PlaceHolder[target=add_6]
#   %iota : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.5), kwargs = {})
#   %sub : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %view : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [512, 1]), kwargs = {})
#   %convert_element_type_1 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %add_1 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1), kwargs = {})
#   %clamp_max : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_1, 255), kwargs = {})
#   %iota_1 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_2 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_1 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, 0.5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0.0), kwargs = {})
#   %convert_element_type_3 : Tensor "i64[512][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %add_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_max_1 : Tensor "i64[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_3, 255), kwargs = {})
#   %_unsafe_index_3 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_4 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub_2 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_3 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_3 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_6 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %sub_5 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_4 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_6 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_4), kwargs = {})
#   %mul_5 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 2.0), kwargs = {})
#   %sub_7 : Tensor "f32[1, 3, 512, 512][786432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_5, 1.0), kwargs = {})
#   return %mul_3,%sub_6,%add_6,%sub_7
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 512) % 512)
    x0 = (xindex % 512)
    x2 = xindex // 262144
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = tl.full([1], 0.0, tl.float32)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 255, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16 - tmp2
    tmp18 = triton_helpers.maximum(tmp17, tmp6)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tmp19 + tmp9
    tmp21 = triton_helpers.minimum(tmp20, tmp11)
    tmp22 = tl.load(in_ptr0 + (tmp21 + 256*tmp12 + 65536*x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (tmp19 + 256*tmp12 + 65536*x2), None, eviction_policy='evict_last')
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19.to(tl.float32)
    tmp26 = tmp18 - tmp25
    tmp27 = triton_helpers.maximum(tmp26, tmp6)
    tmp28 = tl.full([1], 1.0, tl.float32)
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp24 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tl.load(in_ptr0 + (tmp19 + 256*tmp8 + 65536*x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp21 + 256*tmp8 + 65536*x2), None, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp29
    tmp36 = tmp32 + tmp35
    tmp37 = tmp31 - tmp36
    tmp38 = tmp8.to(tl.float32)
    tmp39 = tmp7 - tmp38
    tmp40 = triton_helpers.maximum(tmp39, tmp6)
    tmp41 = triton_helpers.minimum(tmp40, tmp28)
    tmp42 = tmp37 * tmp41
    tmp43 = tmp36 + tmp42
    tmp44 = tl.full([1], 2.0, tl.float32)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp45 - tmp28
    tl.store(in_out_ptr0 + (x3), tmp46, None)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/r7/cr7vgvzs5ehki57y2jomqttqt3yts3zqi2bewzuua74v6llo7wix.py
# Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   mask => full_default
# Graph fragment:
#   %full_default : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_ones_1 = async_compile.triton('triton_poi_fused_ones_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], True, tl.int1)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /work/.inductor_cache_flash/fl/cflzshtpnpj4v2jwddapwbzm3fau6d65owkegwwzbabrwyfdrigy.py
# Topologically Sorted Source Nodes: [new_vector, setitem], Original ATen: [aten.zeros, aten.slice, aten.copy]
# Source node to ATen node mapping:
#   new_vector => full_3
#   setitem => copy, slice_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, 6][6, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %full_3 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_1 : Tensor "f32[1, 6][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%full_3, 1, 0, 6), kwargs = {})
#   %copy : Tensor "f32[1, 6][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %arg3_1), kwargs = {})
#   %slice_scatter_default : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_3, %copy, 1, 0, 6), kwargs = {})
#   return %slice_scatter_default
triton_poi_fused_copy_slice_zeros_2 = async_compile.triton('triton_poi_fused_copy_slice_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_zeros_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 6, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg1_1, (1, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (1, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg3_1, (1, 6), (6, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3, 512, 512), (786432, 262144, 512, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [resized_img, mul, img], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0.run(buf3, arg0_1, 786432, stream=stream0)
        del arg0_1
        buf4 = empty_strided_cuda((1, 3, 512, 512), (786432, 262144, 512, 1), torch.float32)
        buf5 = buf4; del buf4  # reuse
        buf6 = buf5; del buf5  # reuse
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [resized_img_1, mul_1, img_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0.run(buf7, arg1_1, 786432, stream=stream0)
        del arg1_1
        buf8 = empty_strided_cuda((1, 3, 512, 512), (786432, 262144, 512, 1), torch.float32)
        buf9 = buf8; del buf8  # reuse
        buf10 = buf9; del buf9  # reuse
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [resized_img_2, mul_2, img_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0.run(buf11, arg2_1, 786432, stream=stream0)
        del arg2_1
        buf12 = empty_strided_cuda((1, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [mask], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_1.run(buf12, 1, stream=stream0)
        buf13 = empty_strided_cuda((1, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [mask_1], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_1.run(buf13, 1, stream=stream0)
        buf14 = empty_strided_cuda((1, ), (1, ), torch.bool)
        # Topologically Sorted Source Nodes: [mask_2], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_1.run(buf14, 1, stream=stream0)
        buf15 = empty_strided_cuda((1, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [new_vector, setitem], Original ATen: [aten.zeros, aten.slice, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_slice_zeros_2.run(arg3_1, buf15, 32, stream=stream0)
        del arg3_1
    return (buf3, buf7, buf11, buf12, buf13, buf14, buf15, )


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
        partition0_args = [arg0_1, arg1_1, arg2_1, arg3_1]
        del arg0_1, arg1_1, arg2_1, arg3_1
        (buf3, buf7, buf11, buf12, buf13, buf14, buf15) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf3, buf7, buf11, buf12, buf13, buf14, buf15, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
