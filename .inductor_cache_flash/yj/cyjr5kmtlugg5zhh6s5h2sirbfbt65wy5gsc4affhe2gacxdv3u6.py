# AOT ID: ['7_inference']
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



# kernel path: /work/.inductor_cache_flash/ry/cryx5gfyzgchc326wifayqxxlfzk6q275slg3coskvi4niqjjaap.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   x => view
#   x_1 => view_1
#   x_2 => permute
#   x_3 => clone, view_2
#   x_4 => permute_1
#   x_5 => clone_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1024, 768][786432, 768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view : Tensor "bf16[1, 32, 32, 768][786432, 24576, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [1, 32, 32, 768]), kwargs = {})
#   %view_1 : Tensor "bf16[1, 32, 8, 3072][786432, 24576, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 32, 8, 3072]), kwargs = {})
#   %permute : Tensor "bf16[1, 8, 32, 3072][786432, 3072, 24576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "bf16[1, 8, 32, 3072][786432, 98304, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view_2 : Tensor "bf16[1, 8, 8, 12288][786432, 98304, 12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 8, 8, 12288]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 8, 8, 12288][786432, 12288, 98304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 8, 8, 12288][786432, 98304, 12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_permute_view_0 = async_compile.triton('triton_poi_fused__unsafe_view_clone_permute_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '399E50A3AF6A9A1E6899E01CB1A675B878ADF966B91D45205A7FC9E0251DE912', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_permute_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 12288)
    x1 = ((xindex // 12288) % 8)
    x2 = xindex // 98304
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (3072*x1 + 3072*((x0 + 12288*x2) // 98304) + 24576*(x0 // 3072) + 98304*x2 + ((x0 % 3072))), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(arg1_1, (960, 12288), (12288, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 8, 8, 12288), (786432, 98304, 12288, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_permute_view_0.run(arg0_1, buf0, 786432, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((64, 960), (960, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3, x_4, x_5, image_hidden_states], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (64, 12288), (12288, 1), 0), reinterpret_tensor(arg1_1, (12288, 960), (1, 12288), 0), out=buf1)
        del arg1_1
        del buf0
    return (buf1, )


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
        arg0_1, arg1_1 = args
        args.clear()
        partition0_args = [arg0_1, arg1_1]
        del arg0_1, arg1_1
        (buf1,) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf1, (1, 64, 960), (61440, 960, 1), 0), )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((960, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
