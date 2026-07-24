
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_bitwise_and_exp_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2169, 'r0_': 6971889}}
)
@triton.jit
def triton_red_fused__softmax__to_copy_bitwise_and_exp_le_mul_prepare_softmax_online_scalar_tensor_sub_unsqueeze_view_where_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3615
    r0_numel = 241
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 241)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    x3 = xindex
    x1 = xindex // 241
    _tmp13_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp13_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp7 = tl.load(in_ptr2 + (r0_2 + 241*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 & tmp4
        tmp6 = tmp2 & tmp5
        tmp8 = tl.full([1, 1], 0.125, tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])

        _tmp13_max_next, _tmp13_sum_next = triton_helpers.online_softmax_combine(
            _tmp13_max, _tmp13_sum, tmp12, False
        )

        _tmp13_max = tl.where(r0_mask & xmask, _tmp13_max_next, _tmp13_max)
        _tmp13_sum = tl.where(r0_mask & xmask, _tmp13_sum_next, _tmp13_sum)
        tl.store(out_ptr0 + (r0_2 + 241*x0 + 58112*x1), tmp11, r0_mask & xmask)

    tmp13, tmp14 = triton_helpers.online_softmax_reduce(
        _tmp13_max, _tmp13_sum, 1, False)
    tmp13 = tmp13[:, None]
    tmp14 = tmp14[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp15 = tl.load(out_ptr0 + (r0_2 + 241*x0 + 58112*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp15 - tmp13
        tmp17 = libdevice.exp(tmp16)
        tmp18 = (tmp17 / tmp14)
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr3 + (r0_2 + 241*x0 + 58112*x1), tmp19, r0_mask & xmask)
