"""The measured variants of SmolVLA inference, shared by all harnesses.

  original     eager PyTorch (the baseline)
  tuned        eager + our fused-softmax patch (F.softmax interception)
  compiled     torch.compile(default) — Dynamo/Inductor fusion
  compiled-ro  torch.compile(mode="reduce-overhead") — adds CUDA Graphs

Each harness gets (context_manager, infer_fn) from make_infer() so the same
inference path is timed/gated everywhere.
"""
from __future__ import annotations

import contextlib
import time

import torch

from vla.patch_kernels import use_custom_kernels

VARIANTS = ["original", "tuned", "compiled", "compiled-ro"]


def base_infer(policy):
    """The real model forward (chunk prediction), bypassing the action queue."""
    return (getattr(policy, "predict_action_chunk", None)
            or getattr(policy, "select_action", None) or policy.forward)


def make_infer(policy, variant: str):
    """Return (ctx, fn): run fn(obs) inside ctx to execute the variant."""
    base = base_infer(policy)
    if variant == "original":
        return use_custom_kernels(False), base
    if variant == "tuned":
        return use_custom_kernels(True), base
    if variant in ("compiled", "compiled-ro"):
        mode = "reduce-overhead" if variant == "compiled-ro" else None
        return contextlib.nullcontext(), torch.compile(base, mode=mode)
    raise ValueError(f"unknown variant {variant!r}")


def warmup(fn, obs, iters: int = 3):
    """Run fn a few times (compile happens on the first call); returns the
    wall time of the first call so compile cost can be reported."""
    t0 = time.perf_counter()
    fn(obs)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0
    for _ in range(iters - 1):
        fn(obs)
    torch.cuda.synchronize()
    return first_call_s


def dynamo_report() -> dict:
    """Graph/break counters after a compiled run (empty for eager variants)."""
    try:
        from torch._dynamo.utils import counters
        stats = counters.get("stats", {})
        return {
            "graph_breaks": sum(counters.get("graph_break", {}).values()),
            "unique_graphs": stats.get("unique_graphs", 0),
        }
    except Exception:
        return {}
