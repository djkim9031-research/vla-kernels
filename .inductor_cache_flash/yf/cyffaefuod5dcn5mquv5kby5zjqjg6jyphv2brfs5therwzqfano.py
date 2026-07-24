
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 224000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x3 = xindex
    x2 = xindex // 320
    tmp0 = x0
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp2, tmp6, 0)
    tmp8 = 1 + x2
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full([1], 0.03125, tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full([1], 10000.0, tl.float32)
    tmp18 = libdevice.pow(tmp17, tmp16)
    tmp19 = (tmp12 / tmp18)
    tmp20 = tl_math.cos(tmp19)
    tmp21 = tmp4 * tmp20
    tmp22 = tl.load(in_ptr0 + (32 + x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl_math.sin(tmp19)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 - tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp2, tmp26, tmp27)
    tmp29 = tl.full([1], float("nan"), tl.float32)
    tmp30 = tl.where(tmp2, tmp28, tmp29)
    tmp31 = tmp0 >= tmp1
    tmp32 = tl.load(in_ptr0 + (x3), tmp31 & xmask, other=0.0).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.load(in_ptr1 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp36 = tl.where(tmp31, tmp35, 0)
    tmp37 = 1 + x2
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1], 1, tl.int64)
    tmp40 = tmp38 - tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = (-32) + x0
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.full([1], 0.03125, tl.float32)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full([1], 10000.0, tl.float32)
    tmp47 = libdevice.pow(tmp46, tmp45)
    tmp48 = (tmp41 / tmp47)
    tmp49 = tl_math.cos(tmp48)
    tmp50 = tmp33 * tmp49
    tmp51 = tl.load(in_ptr0 + ((-32) + x3), tmp31 & xmask, other=0.0).to(tl.float32)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tl_math.sin(tmp48)
    tmp54 = tmp52 * tmp53
    tmp55 = tmp50 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp31, tmp55, tmp56)
    tmp58 = tl.where(tmp31, tmp57, tmp30)
    tl.store(in_out_ptr0 + (x3), tmp58, xmask)
