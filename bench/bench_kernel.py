#!/usr/bin/env python3
"""Op-agnostic kernel benchmark: correctness gate + latency + achieved bandwidth.

Every new kernel registers a (custom_fn, torch_reference, bytes_moved) entry and
gets latency, GB/s, and speedup-vs-torch for free. Currently wired for softmax.

Example:
    python3 bench/bench_kernel.py --op softmax --dtypes fp32,fp16 --out results/softmax.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def cuda_time_ms(fn, iters: int, warmup: int = 25) -> float:
    """Median-ish latency via CUDA events (returns mean over `iters`)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_softmax(args):
    from kernels.softmax import softmax, VERSIONS

    # library baselines (cuDNN softmax, CUB-primitive softmax); skip gracefully
    baselines = {}
    try:
        from kernels.softmax import softmax_cudnn, softmax_cub
        baselines = {"cudnn": softmax_cudnn, "cub": softmax_cub}
    except Exception as e:  # missing cudnn wheel / CCCL headers
        print(f"note: library baselines unavailable ({e})")

    # softmax reads x and writes y once => 2 * numel * itemsize bytes (lower bound)
    rows_cols = [(1024, 128), (2048, 512), (512, 2048), (256, 4096), (64, 16384)]
    variants = {**{v: (lambda x, v=v: softmax(x, v)) for v in VERSIONS},
                **{n: (lambda x, f=f: f(x)) for n, f in baselines.items()}}
    rows = []
    for dt_name in args.dtypes.split(","):
        dt = DTYPES[dt_name]
        for (R, C) in rows_cols:
            x = torch.randn(R, C, device="cuda", dtype=dt)
            ref = torch.softmax(x, dim=-1)
            bytes_moved = 2 * x.numel() * x.element_size()

            # torch baseline
            t_torch = cuda_time_ms(lambda: torch.softmax(x, dim=-1), args.iters)

            for name, fn in variants.items():
                out = fn(x)
                max_err = (out.float() - ref.float()).abs().max().item()
                t = cuda_time_ms(lambda: fn(x), args.iters)
                gbps = bytes_moved / (t * 1e-3) / 1e9
                row = dict(op="softmax", dtype=dt_name, rows=R, cols=C, version=name,
                           ms=round(t, 5), gbps=round(gbps, 1),
                           speedup_vs_torch=round(t_torch / t, 3),
                           max_abs_err=f"{max_err:.2e}")
                rows.append(row)
                print(f"{dt_name} [{R:>5}x{C:<6}] {name:>5}: {t:8.4f} ms  "
                      f"{gbps:7.1f} GB/s  x{row['speedup_vs_torch']:<5} torch={t_torch:.4f}ms  err={max_err:.1e}")
    return rows


OPS = {"softmax": bench_softmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="softmax", choices=list(OPS))
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--once", action="store_true",
                    help="single forward of the best version (for ncu capture)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    if args.once:  # minimal work for profiler capture
        from kernels.softmax import softmax, BEST
        x = torch.randn(2048, 2048, device="cuda")
        softmax(x, BEST)
        torch.cuda.synchronize()
        return

    rows = OPS[args.op](args)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
