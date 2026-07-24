
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
