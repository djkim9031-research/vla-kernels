
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_div_mm_mul_pow_sin_slice_split_sub_unsqueeze_view_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 11, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6940800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_div_mm_mul_pow_sin_slice_split_sub_unsqueeze_view_32(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x3 = xindex
    x2 = xindex // 320
    tmp0 = x0
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (-32) + x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 0.03125, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full([1], 10000.0, tl.float32)
    tmp14 = libdevice.pow(tmp13, tmp12)
    tmp15 = (tmp8 / tmp14)
    tmp16 = tl_math.cos(tmp15)
    tmp17 = tmp4 * tmp16
    tmp18 = tl.load(in_ptr0 + ((-32) + x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl_math.sin(tmp15)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp17 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tmp0 < tmp1
    tmp26 = tl.load(in_ptr0 + (x3), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full([1], 1, tl.int64)
    tmp30 = tmp28 - tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = x0
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.full([1], 0.03125, tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.full([1], 10000.0, tl.float32)
    tmp37 = libdevice.pow(tmp36, tmp35)
    tmp38 = (tmp31 / tmp37)
    tmp39 = tl_math.cos(tmp38)
    tmp40 = tmp27 * tmp39
    tmp41 = tl.load(in_ptr0 + (32 + x3), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl_math.sin(tmp38)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp40 - tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.full([1], float("nan"), tl.float32)
    tmp49 = tl.where(tmp25, tmp47, tmp48)
    tmp50 = tl.where(tmp2, tmp24, tmp49)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp51, xmask)
    tl.store(out_ptr2 + (x3), tmp52, xmask)
    tl.store(out_ptr3 + (x3), tmp52, xmask)
    tl.store(out_ptr4 + (x3), tmp52, xmask)
    tl.store(out_ptr5 + (x3), tmp52, xmask)
    tl.store(out_ptr6 + (x3), tmp52, xmask)
    tl.store(out_ptr7 + (x3), tmp52, xmask)
    tl.store(out_ptr8 + (x3), tmp52, xmask)
    tl.store(out_ptr9 + (x3), tmp52, xmask)
    tl.store(out_ptr10 + (x3), tmp52, xmask)
    tl.store(out_ptr11 + (x3), tmp52, xmask)
