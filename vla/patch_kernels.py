"""Swap our hand-written kernels into a model at runtime (no model edits).

`use_custom_kernels()` is a context manager that monkeypatches
``torch.nn.functional.softmax`` so last-dim softmax calls on CUDA tensors route
through our fused kernel, with an automatic fallback to the stock op for any
shape/dtype we don't handle.

Note on scope: SmolVLA's transformer attention uses the fused
``scaled_dot_product_attention`` (SDPA), which does NOT call F.softmax
separately — so this patch targets the *explicit* softmax calls (e.g. action /
logit normalization). Replacing the attention softmax itself is the job of the
fused-attention kernel on the roadmap (kernels/attention/), which swaps SDPA.
This separation keeps the speed/accuracy attribution clean: gains reported for
the softmax patch come only from the ops it actually replaces.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from kernels.softmax import softmax as fused_softmax

_orig_softmax = F.softmax


def _patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
    last = dim in (-1, input.dim() - 1) if dim is not None else False
    if (last and input.is_cuda and input.is_contiguous()
            and input.dtype in (torch.float32, torch.float16) and dtype is None):
        x2d = input.reshape(-1, input.shape[-1])
        return fused_softmax(x2d, "auto").reshape(input.shape)
    return _orig_softmax(input, dim=dim, dtype=dtype)


@contextlib.contextmanager
def use_custom_kernels(enabled: bool = True):
    """Within this block, eligible F.softmax calls use the fused CUDA kernel."""
    if not enabled:
        yield
        return
    F.softmax = _patched_softmax
    try:
        yield
    finally:
        F.softmax = _orig_softmax
