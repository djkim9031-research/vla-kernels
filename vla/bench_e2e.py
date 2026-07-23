"""End-to-end SmolVLA chunk-inference latency on Thor, per variant.

Reports p50/p99 latency and chunk rate for any variant in vla/variants.py
(eager, kernel-patched, torch.compile, compile+CUDA-graphs). Pair with
tegrastats for power.

Usage (in Docker):
    python3 vla/bench_e2e.py --variant original --out results/e2e_original.csv
    python3 vla/bench_e2e.py --variant compiled --out results/e2e_compiled.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics

import torch

from vla.load_smolvla import load_policy, dummy_observation
from vla.variants import VARIANTS, make_infer, warmup, dynamo_report


@torch.no_grad()
def latencies_ms(fn, obs, iters: int):
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(obs)
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="original")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    policy = load_policy()
    obs = dummy_observation(policy)

    ctx, fn = make_infer(policy, args.variant)
    with ctx, torch.no_grad():
        first_s = warmup(fn, obs, args.warmup)
        lat = latencies_ms(fn, obs, args.iters)

    p50, p99, mean = pct(lat, 0.50), pct(lat, 0.99), statistics.mean(lat)
    row = dict(variant=args.variant, p50_ms=round(p50, 3), p99_ms=round(p99, 3),
               mean_ms=round(mean, 3), hz=round(1000.0 / p50, 1),
               first_call_s=round(first_s, 1), **dynamo_report())
    print(row)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
