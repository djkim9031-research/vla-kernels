
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 288000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_cos_expand_linspace_mul_pow_reciprocal_sin_unsqueeze_view_57(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 720)
    x1 = xindex // 720
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 360, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1], 180.0, tl.float32)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5.to(tl.float64)
    tmp10 = tl.full([1], 0.002785515320334262, tl.float64)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.0, tl.float64)
    tmp13 = tmp11 + tmp12
    tmp14 = 359 + (-1)*(x0)
    tmp15 = tmp14.to(tl.float64)
    tmp16 = tmp15 * tmp10
    tmp17 = tl.full([1], 1.0, tl.float64)
    tmp18 = tmp17 - tmp16
    tmp19 = tl.where(tmp8, tmp13, tmp18)
    tmp20 = tl.full([1], 1000.0, tl.float64)
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = tl.full([1], 0.004, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full([1], 1, tl.int32)
    tmp25 = (tmp24 / tmp23)
    tmp26 = tmp25 * tmp17
    tmp27 = tl.full([1], 2.0, tl.float64)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.full([1], 3.141592653589793, tl.float64)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.full([1], 0.30000001192092896, tl.float64)
    tmp32 = tmp30 * tmp31
    tmp33 = libdevice.sin(tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp4, tmp33, tmp34)
    tmp36 = tmp0 >= tmp3
    tmp37 = tl.full([1], 720, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = (-360) + x0
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.full([1], 180.0, tl.float32)
    tmp42 = tmp40 < tmp41
    tmp43 = tmp39.to(tl.float64)
    tmp44 = tl.full([1], 0.002785515320334262, tl.float64)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full([1], 0.0, tl.float64)
    tmp47 = tmp45 + tmp46
    tmp48 = 359 + (-1)*((-360) + x0)
    tmp49 = tmp48.to(tl.float64)
    tmp50 = tmp49 * tmp44
    tmp51 = tl.full([1], 1.0, tl.float64)
    tmp52 = tmp51 - tmp50
    tmp53 = tl.where(tmp42, tmp47, tmp52)
    tmp54 = tl.full([1], 1000.0, tl.float64)
    tmp55 = libdevice.pow(tmp54, tmp53)
    tmp56 = tl.full([1], 0.004, tl.float64)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.full([1], 1, tl.int32)
    tmp59 = (tmp58 / tmp57)
    tmp60 = tmp59 * tmp51
    tmp61 = tl.full([1], 2.0, tl.float64)
    tmp62 = tmp60 * tmp61
    tmp63 = tl.full([1], 3.141592653589793, tl.float64)
    tmp64 = tmp62 * tmp63
    tmp65 = tl.full([1], 0.30000001192092896, tl.float64)
    tmp66 = tmp64 * tmp65
    tmp67 = libdevice.cos(tmp66)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp36, tmp67, tmp68)
    tmp70 = tl.where(tmp4, tmp35, tmp69)
    tmp71 = tmp70.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 1440*x1), tmp71, xmask)
