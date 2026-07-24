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


@contextlib.contextmanager
def _route_eager_to_sdpa():
    """Route the expert/prefix eager attention (176 sites) to F.sdpa.

    lerobot's SmolVLMWithExpertModel implements attention manually:
    fp32-upcast QK^T -> where(mask, scores, big_neg) -> softmax -> PV, with a
    (B, Sq, Sk) bool mask. Op-level measurement showed F.sdpa 3-6x faster at
    these exact shapes (the mem-efficient backend accepts bool masks; it
    accumulates in fp32 internally, replacing the explicit upcast). GQA k/v
    expansion is kept explicit — sdpa's enable_gqa falls back to the math
    backend when a mask is present.

    Applied to compiled variants only, so `original` remains the untouched
    eager baseline.
    """
    try:
        from lerobot.policies.smolvla import smolvlm_with_expert as swe
    except ImportError:
        yield
        return
    import torch.nn.functional as F

    def sdpa_forward(self, attention_mask, batch_size, head_dim,
                     query_states, key_states, value_states):
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads
        groups = n_heads // n_kv
        seq_k = key_states.shape[1]
        if groups > 1:  # GQA: expand k/v to full head count, like the eager path
            key_states = key_states[:, :, :, None, :].expand(
                batch_size, seq_k, n_kv, groups, head_dim).reshape(
                batch_size, seq_k, n_heads, head_dim)
            value_states = value_states[:, :, :, None, :].expand(
                batch_size, seq_k, n_kv, groups, head_dim).reshape(
                batch_size, seq_k, n_heads, head_dim)
        # callers mix dtypes (bf16 q, fp32 cached k/v) — the eager path upcast
        # everything to fp32; sdpa needs uniform dtype, unify at q's
        q = query_states.transpose(1, 2)
        k = key_states.to(query_states.dtype).transpose(1, 2)
        v = value_states.to(query_states.dtype).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask[:, None, :, :],
            scale=head_dim ** -0.5)
        return out.transpose(1, 2).reshape(batch_size, -1, n_heads * head_dim)

    cls = swe.SmolVLMWithExpertModel
    orig = cls.eager_attention_forward
    cls.eager_attention_forward = sdpa_forward
    try:
        yield
    finally:
        cls.eager_attention_forward = orig


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
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_eager_to_sdpa()), \
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
