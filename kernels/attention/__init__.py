"""Fused scaled-dot-product attention (non-causal, head_dim 64).

JIT-built like kernels.softmax; registered with the dispatcher as
vlak::fused_attention so it traces as a single node under torch.compile /
torch.export — the pattern proven with vlak::softmax.
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def _ext():
    ext = load(
        name="vlak_attention",
        sources=[os.path.join(_THIS_DIR, "fused_attention.cu")],
        extra_cuda_cflags=["-O3", "-arch=native", "--use_fast_math"],
        verbose=bool(int(os.environ.get("VLAK_VERBOSE", "0"))),
    )
    torch.library.register_fake("vlak::fused_attention")(
        lambda q, k, v, scale=-1.0: torch.empty_like(q))
    return ext


_registered = False


def _ensure_registered():
    global _registered
    if not _registered:
        _ext()
        _registered = True


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float | None = None) -> torch.Tensor:
    """Non-causal SDPA for (B, H, M, 64) tensors; scores never touch HBM."""
    _ensure_registered()
    return torch.ops.vlak.fused_attention(q, k, v,
                                          -1.0 if scale is None else scale)
