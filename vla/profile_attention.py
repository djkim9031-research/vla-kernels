"""Phase D step 1: size the attention budget inside SmolVLA inference.

Two complementary measurements:

  --variant compiled-ro   torch.profiler over the compiled model; kernels
                          classified by name family (fused softmax, SDPA/fmha,
                          GEMM, generated elementwise, other). GEMMs cannot be
                          attributed to attention-vs-MLP by name alone.
  --variant original      eager model with every *Attention module's forward
                          wrapped in a profiler range -> clean attribution,
                          including the GEMM split. cuBLAS calls are identical
                          extern kernels in both worlds, so the eager
                          attention-GEMM share transfers to the compiled run.

Usage (in Docker):
    python3 vla/profile_attention.py --variant original
    python3 vla/profile_attention.py --variant compiled-ro
"""
from __future__ import annotations

import argparse
import collections
import re

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from vla.load_smolvla import load_policy, dummy_observation
from vla.variants import make_infer, warmup

FAMILIES = [
    ("attn-softmax (triton fused)", re.compile(r"softmax", re.I)),
    ("vision SDPA (fmha/flash)", re.compile(r"fmha|flash|attention", re.I)),
    ("GEMM (cublas/cutlass)", re.compile(r"gemm|cutlass|nvjet|s16816|cublas", re.I)),
    ("generated elementwise/reduction", re.compile(r"^triton_")),
    ("elementwise (aten)", re.compile(r"elementwise|vectorized|reduce_kernel", re.I)),
]


def classify(name: str) -> str:
    for fam, pat in FAMILIES:
        if pat.search(name):
            return fam
    return "other"


def wrap_attention_modules(policy):
    """Wrap top-level attention modules in a profiler range (eager only).

    Only outermost matches are wrapped — wrapping a child of an already
    wrapped module would nest identical ranges and double-count device time.
    """
    wrapped_names: list[str] = []
    for name, mod in policy.named_modules():
        cls = type(mod).__name__
        if "Attention" not in cls and "attention" != name.split(".")[-1]:
            continue
        if any(name == w or name.startswith(w + ".") for w in wrapped_names):
            continue  # ancestor already wrapped
        orig = mod.forward

        def wrapped(*a, _orig=orig, **kw):
            with record_function("ATTN_REGION"):
                return _orig(*a, **kw)

        mod.forward = wrapped
        wrapped_names.append(name)
    return len(wrapped_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "compiled", "compiled-ro"],
                    default="compiled-ro")
    ap.add_argument("--steps", type=int, default=3, help="profiled inferences")
    args = ap.parse_args()

    policy = load_policy()
    obs = dummy_observation(policy)

    if args.variant == "original":
        nmod = wrap_attention_modules(policy)
        print(f"wrapped {nmod} attention modules in profiler ranges")

    ctx, fn = make_infer(policy, args.variant)
    with ctx, torch.no_grad():
        warmup(fn, obs, 5)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                fn(obs)
            torch.cuda.synchronize()

    evts = prof.key_averages()
    # SELF device time only: host ops (aten::linear, compiled-region markers)
    # aggregate their children's device time in *_total, which double-counts
    kernels = [e for e in evts if e.self_device_time_total > 0
               and e.key != "ATTN_REGION"
               and "memcpy" not in e.key.lower() and "memset" not in e.key.lower()]
    total_us = sum(e.self_device_time_total for e in kernels)

    fam_us = collections.Counter()
    for e in kernels:
        fam_us[classify(e.key)] += e.self_device_time_total

    per_inf = total_us / args.steps / 1000.0
    print(f"\n=== {args.variant}: {per_inf:.1f} ms GPU time per inference "
          f"(sum over {args.steps} steps) ===")
    print(f"{'family':38} {'ms/inf':>8}  {'share':>6}")
    for fam, us in fam_us.most_common():
        print(f"{fam:38} {us/args.steps/1000:8.2f}  {us/total_us*100:5.1f}%")

    if args.variant == "original":
        attn = [e for e in evts if e.key == "ATTN_REGION"]
        if attn:
            attn_us = sum(e.device_time_total for e in attn)
            print(f"\nATTN_REGION device time: {attn_us/args.steps/1000:.2f} ms/inf "
                  f"({attn_us/total_us*100:.1f}% of GPU time)")

    print("\ntop 12 kernels (self device time):")
    for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:12]:
        print(f"  {e.self_device_time_total/args.steps/1000:7.2f} ms  x{e.count//args.steps:<4} "
              f"{e.key[:90]}")


if __name__ == "__main__":
    main()
