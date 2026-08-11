"""Paired v4 vs v4s kernel microbench at the real expert-site shapes.

Locked clocks assumed (bench/rig.sh). 7 reps x 200 iters, CUDA events,
kernel order alternated per rep to cancel order bias; medians reported.
"""
import torch

from kernels.attention import fused_attention_gqa_v4 as v4
from kernels.attention import fused_attention_gqa_v4s as v4s

torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
REPS, ITERS = 7, 200

SITES = [
    ("cross 50x241", 241),
    ("self  50x291", 291),
]


def time_kernel(fn, q, k, v, P, ds, de):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(20):
        fn(q, k, v, prefix_len=P, dead_start=ds, dead_end=de)
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        fn(q, k, v, prefix_len=P, dead_start=ds, dead_end=de)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / ITERS  # us/call


for name, N in SITES:
    q = torch.randn(1, 50, 15, 64, device=dev, dtype=dt)
    k = torch.randn(1, N, 5, 64, device=dev, dtype=dt)
    v = torch.randn(1, N, 5, 64, device=dev, dtype=dt)
    P, ds, de = 241, 196, 240
    t4, t4s = [], []
    for r in range(REPS):
        pair = [(v4, t4), (v4s, t4s)] if r % 2 == 0 else [(v4s, t4s), (v4, t4)]
        for fn, acc in pair:
            acc.append(time_kernel(fn, q, k, v, P, ds, de))
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"{name}:  v4 {med(t4):6.2f} us  (all: {[f'{x:.2f}' for x in t4]})")
    print(f"{name}:  v4s {med(t4s):6.2f} us  (all: {[f'{x:.2f}' for x in t4s]})")
    print(f"{name}:  delta {med(t4s) - med(t4):+.2f} us "
          f"({(med(t4s) / med(t4) - 1) * 100:+.1f}%)")
