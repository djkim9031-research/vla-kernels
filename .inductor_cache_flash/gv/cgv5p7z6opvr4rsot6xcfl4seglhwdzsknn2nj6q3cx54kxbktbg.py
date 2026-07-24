
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
