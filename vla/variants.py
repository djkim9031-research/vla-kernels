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


def _sdpa_flash_first():
    """Pin SDPA backend priority: flash first, mem-efficient as fallback.

    Profiling (results/attention_budget.md) showed Inductor's lowering picks
    fmha_cutlassF (11.24 ms/inf for the 36 vision calls) where eager's
    dispatcher picks flash_fwd (3.55 ms) — a 3.2x backend regression. The
    context must be active during tracing, so make_infer folds it into every
    variant's ctx (eager already picks flash; pinning keeps variants uniform).
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # global switches too: the thread-local context alone is not always
    # consulted by Inductor's lowering (and is not part of its cache key)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)  # last-resort fallback only

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION])


@contextlib.contextmanager
def _vision_mask_none():
    """Drop the vision tower's attention mask so flash stays eligible.

    lerobot passes patch_attention_mask=None; SmolVLM then manufactures an
    all-ones mask (torch.ones two lines above the call) and
    create_bidirectional_mask 4D-ifies it. Eager's all-ones shortcut returns
    None (flash runs); under tracing that data-dependent check is skipped and
    the materialized mask forces the efficient backend (3.2x slower here).
    With fixed 256x256 cameras and no padding the mask is all-ones by
    construction, so returning None is semantics-preserving for this
    deployment.
    """
    try:
        from transformers.models.smolvlm import modeling_smolvlm as mm
    except ImportError:
        yield
        return
    orig = mm.create_bidirectional_mask
    mm.create_bidirectional_mask = lambda *a, **kw: None
    try:
        yield
    finally:
        mm.create_bidirectional_mask = orig


def _stack(*ctxs):
    es = contextlib.ExitStack()

    @contextlib.contextmanager
    def combined():
        with es:
            for c in ctxs:
                es.enter_context(c)
            yield

    return combined()


def base_infer(policy):
    """The real model forward (chunk prediction), bypassing the action queue."""
    return (getattr(policy, "predict_action_chunk", None)
            or getattr(policy, "select_action", None) or policy.forward)


def make_infer(policy, variant: str):
    """Return (ctx, fn): run fn(obs) inside ctx to execute the variant."""
    base = base_infer(policy)
    if variant == "original":
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      use_custom_kernels(False)), base
    if variant == "tuned":
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      use_custom_kernels(True)), base
    if variant in ("compiled", "compiled-ro"):
        mode = "reduce-overhead" if variant == "compiled-ro" else None
        return _stack(_sdpa_flash_first(), _vision_mask_none()), \
            torch.compile(base, mode=mode)
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
