
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 672000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_copy_cos_cumsum_div_mul_pow_sin_slice_split_sub_unsqueeze_view_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x3 = xindex
    x2 = xindex // 960
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
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp14 = tl.where(tmp2, tmp13, 0)
    tmp15 = tmp11 - tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = x0
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1], 0.03125, tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.full([1], 10000.0, tl.float32)
    tmp22 = libdevice.pow(tmp21, tmp20)
    tmp23 = (tmp16 / tmp22)
    tmp24 = tl_math.cos(tmp23)
    tmp25 = tmp4 * tmp24
    tmp26 = tl.load(in_ptr0 + (32 + x3), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl_math.sin(tmp23)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 - tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp2, tmp30, tmp31)
    tmp33 = tl.full([1], float("nan"), tl.float32)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp35 = tmp0 >= tmp1
    tmp36 = tl.load(in_ptr0 + (x3), tmp35 & xmask, other=0.0).to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tl.load(in_ptr1 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp40 = tl.where(tmp35, tmp39, 0)
    tmp41 = 1 + x2
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1], 1, tl.int64)
    tmp44 = tmp42 - tmp43
    tmp45 = tl.load(in_ptr2 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp47 = tl.where(tmp35, tmp46, 0)
    tmp48 = tmp44 - tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = (-32) + x0
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tl.full([1], 0.03125, tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tl.full([1], 10000.0, tl.float32)
    tmp55 = libdevice.pow(tmp54, tmp53)
    tmp56 = (tmp49 / tmp55)
    tmp57 = tl_math.cos(tmp56)
    tmp58 = tmp37 * tmp57
    tmp59 = tl.load(in_ptr0 + ((-32) + x3), tmp35 & xmask, other=0.0).to(tl.float32)
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tl_math.sin(tmp56)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp58 + tmp62
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp35, tmp63, tmp64)
    tmp66 = tl.where(tmp35, tmp65, tmp34)
    tl.store(in_out_ptr0 + (x3), tmp66, xmask)
