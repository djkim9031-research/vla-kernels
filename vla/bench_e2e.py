"""End-to-end SmolVLA inference latency on Thor: original vs. kernel-tuned.

Reports p50/p99 single-inference latency and achievable control-loop Hz. Pair
with tegrastats (run `tegrastats` in another shell) for the TFLOP/s-per-watt
column in the README table.

Usage (in Docker):
    python3 vla/bench_e2e.py --variant original --out results/e2e_original.csv
    python3 vla/bench_e2e.py --variant tuned    --out results/e2e_tuned.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics

import torch

from vla.load_smolvla import load_policy, dummy_observation
from vla.patch_kernels import use_custom_kernels
from vla.eval_accuracy import get_action


@torch.no_grad()
def latencies_ms(policy, obs, iters: int, warmup: int = 20):
    for _ in range(warmup):
        get_action(policy, obs)
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        get_action(policy, obs)
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "tuned"], default="original")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    policy = load_policy()
    obs = dummy_observation(policy)

    with use_custom_kernels(args.variant == "tuned"):
        lat = latencies_ms(policy, obs, args.iters)

    p50, p99, mean = pct(lat, 0.50), pct(lat, 0.99), statistics.mean(lat)
    hz = 1000.0 / p50
    row = dict(variant=args.variant, p50_ms=round(p50, 3), p99_ms=round(p99, 3),
               mean_ms=round(mean, 3), hz=round(hz, 1))
    print(row)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
