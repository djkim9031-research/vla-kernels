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
    # the real kernel contiguous()-izes and returns a CONTIGUOUS tensor; the
    # fake must promise the same strides (empty_like would inherit the input's
    # view strides and make Inductor plan around a layout we never produce)
    torch.library.register_fake("vlak::fused_attention")(
        lambda q, k, v, scale=-1.0, attn_mask=None, prefix_len=-1:
            torch.empty(q.shape, dtype=q.dtype, device=q.device))
    torch.library.register_fake("vlak::fused_attention_gqa")(
        lambda q, k, v, scale=-1.0, prefix_len=-1, dead_start=0, dead_end=0:
            torch.empty(q.shape, dtype=q.dtype, device=q.device))
    return ext


_registered = False


def _ensure_registered():
    global _registered
    if not _registered:
        _ext()
        _registered = True


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float | None = None,
                    attn_mask: torch.Tensor | None = None,
                    prefix_len: int | None = None) -> torch.Tensor:
    """SDPA for (B, H, M, 64) tensors; scores never touch HBM.

    Masking, in preference order:
      prefix_len=P  the SmolVLA pattern as ARITHMETIC (no mask tensor at all):
                    col visible iff col < P or col <= (N - M) + row.
                    Fully-visible tiles take a branchless fast path; invisible
                    tiles are skipped outright. bf16 tensor-core path only.
      attn_mask     generic (B, M, N) bool, broadcast over heads.
      neither       unmasked (fast path everywhere).
    """
    _ensure_registered()
    return torch.ops.vlak.fused_attention(
        q, k, v, -1.0 if scale is None else scale, attn_mask,
        -1 if prefix_len is None else prefix_len)


def fused_attention_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float | None = None,
                        prefix_len: int | None = None,
                        dead_start: int = 0,
                        dead_end: int = 0) -> torch.Tensor:
    """GQA-native SDPA over the model's projection layouts.

    q is (B, M, H, 64); k/v are (B, N, H_kv, 64) with H a multiple of H_kv —
    consumed at their natural strides (no transpose/contiguous/expand). The
    mask is arithmetic: visible(r, c) = c < prefix_len ? (c < dead_start or
    c >= dead_end) : c <= (N - M) + r. Returns (B, M, H, 64) contiguous, so
    .reshape(B, M, H*64) is a view. bf16 only.
    """
    _ensure_registered()
    return torch.ops.vlak.fused_attention_gqa(
        q, k, v, -1.0 if scale is None else scale,
        -1 if prefix_len is None else prefix_len, dead_start, dead_end)
