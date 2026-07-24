
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 200, 'r0_': 1746441}}
)
@triton.jit
def triton_per_fused__softmax__to_copy_bitwise_and_cat_exp_expand_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_16(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 750
    r0_numel = 291
    R0_BLOCK: tl.constexpr = 512
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
    r0_2 = r0_index
    x0 = (xindex % 50)
    x3 = xindex
    x1 = xindex // 50
    tmp17 = tl.load(in_ptr2 + (r0_2 + 291*x3), r0_mask & xmask, other=0.0)
    tmp0 = r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 241, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 291, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (tl.broadcast_to((-241) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 <= tmp10
    tmp12 = tl.full([1, 1], True, tl.int1)
    tmp13 = tmp11 & tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tl.full([1, 1], 0.125, tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(r0_mask & xmask, tmp24, float("-inf"))
    tmp27 = triton_helpers.max2(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tmp22 - tmp27
    tmp29 = libdevice.exp(tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(r0_mask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None].to(tl.float32)
    tmp34 = tmp21 - tmp27
    tmp35 = libdevice.exp(tmp34)
    tmp36 = (tmp35 / tmp33)
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 291*x0 + 14592*x1), tmp37, r0_mask & xmask)
