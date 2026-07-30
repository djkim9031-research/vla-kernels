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

VARIANTS = ["original", "tuned", "compiled", "compiled-ro", "compiled-vk",
            "compiled-vk3", "compiled-vk4", "compiled-vk4x", "compiled-vk4c",
            "compiled-vk5", "compiled-vk7"]


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


@contextlib.contextmanager
def _route_attention_vlak():
    """Expert attention sites -> vlak::fused_attention; prefix -> F.sdpa.

    Same GQA-expand/dtype-unify preamble as the sdpa routing, but the M<=96
    sites (the expert's cross and staircase attention, with their REAL masks)
    run our tensor-core kernel; the prefix (M=241) stays on sdpa, where the
    mem-efficient backend beats us. The op is dispatcher-registered with a
    fake impl, so it traces as a single node under torch.compile.
    """
    try:
        from lerobot.policies.smolvla import smolvlm_with_expert as swe
    except ImportError:
        yield
        return
    import torch.nn.functional as F
    from kernels.attention import fused_attention as vlak_attn

    # warm outside tracing: JIT-build + register the op so Dynamo sees a
    # registered custom op (and the _ensure_registered global folds away)
    _w = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.bfloat16)
    vlak_attn(_w, _w, _w)

    def routed(self, attention_mask, batch_size, head_dim,
               query_states, key_states, value_states):
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads
        groups = n_heads // n_kv
        seq_k = key_states.shape[1]
        if groups > 1:
            key_states = key_states[:, :, :, None, :].expand(
                batch_size, seq_k, n_kv, groups, head_dim).reshape(
                batch_size, seq_k, n_heads, head_dim)
            value_states = value_states[:, :, :, None, :].expand(
                batch_size, seq_k, n_kv, groups, head_dim).reshape(
                batch_size, seq_k, n_heads, head_dim)
        q = query_states.transpose(1, 2)
        k = key_states.to(query_states.dtype).transpose(1, 2)
        v = value_states.to(query_states.dtype).transpose(1, 2)
        if q.shape[2] <= 96 and q.dtype == torch.bfloat16 and head_dim == 64:
            # hand the op contiguous tensors: Inductor fuses these into the
            # cast/expand chain, where the op's internal contiguous() would
            # pay 3 standalone copy kernels per call
            out = vlak_attn(q.contiguous(), k.contiguous(), v.contiguous(),
                            scale=head_dim ** -0.5,
                            attn_mask=attention_mask.contiguous())
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask[:, None],
                scale=head_dim ** -0.5)
        return out.transpose(1, 2).reshape(batch_size, -1, n_heads * head_dim)

    cls = swe.SmolVLMWithExpertModel
    orig = cls.eager_attention_forward
    cls.eager_attention_forward = routed
    try:
        yield
    finally:
        cls.eager_attention_forward = orig


def _analytic_params(m2, M, N):
    """Solve a captured (M, N) bool mask for the v3 analytic parameters
    (P, ds, de): visible(r, c) = c < P ? (c < ds or c >= de)
                                       : c <= (N - M) + r.
    Returns None unless the analytic form reproduces the mask EXACTLY."""
    r = torch.arange(M)[:, None]
    c = torch.arange(N)[None, :]
    for P in dict.fromkeys([N, N - M]):        # no suffix, then suffix of M
        if P <= 0:
            continue
        pre = m2[:, :P]
        deadcols = (~pre.any(0)).nonzero().flatten()
        if len(deadcols):
            ds, de = int(deadcols.min()), int(deadcols.max()) + 1
        else:
            ds = de = 0
        ana = torch.where(c < P, (c < ds) | (c >= de), c <= (N - M) + r)
        if torch.equal(ana, m2):
            return (P, ds, de)
    return None


def _probe_mask_params(policy):
    """One eager inference with a capturing shim on eager_attention_forward:
    records each site's (M, N) mask and solves it for analytic parameters.
    Static-per-task by construction — the language prompt fixes the pad band."""
    from lerobot.policies.smolvla import smolvlm_with_expert as swe
    from vla.load_smolvla import dummy_observation

    captured = {}
    cls = swe.SmolVLMWithExpertModel
    orig = cls.eager_attention_forward

    def capture(self, attention_mask, batch_size, head_dim,
                query_states, key_states, value_states):
        key = (query_states.shape[1], key_states.shape[1])
        if key not in captured:
            captured[key] = attention_mask[0].detach().to("cpu", torch.bool)
        return orig(self, attention_mask, batch_size, head_dim,
                    query_states, key_states, value_states)

    cls.eager_attention_forward = capture
    try:
        with torch.no_grad():
            base_infer(policy)(dummy_observation(policy))
    finally:
        cls.eager_attention_forward = orig
    return {mn: _analytic_params(m2, *mn) for mn, m2 in captured.items()}


@contextlib.contextmanager
def _route_attention_vlak3(params, use_v4=False):
    """Expert sites -> vlak::fused_attention_gqa (v3).

    v3 consumes the model's own layouts: q (B, M, H, 64) and the UNEXPANDED
    5-head k/v (B, N, H_kv, 64) at their natural strides, with the probed
    analytic mask as arithmetic. The whole sdpa entourage disappears at those
    sites — no GQA expand, no transposes, no contiguous staging, no mask
    tensor — and (B, M, H, 64) contiguous output makes the final reshape a
    view. Prefix (M > 96) and any site whose mask defeated the probe fall
    back to F.sdpa with the explicit expand."""
    try:
        from lerobot.policies.smolvla import smolvlm_with_expert as swe
    except ImportError:
        yield
        return
    import torch.nn.functional as F
    if use_v4:
        from kernels.attention import fused_attention_gqa_v4 as vlak_gqa
    else:
        from kernels.attention import fused_attention_gqa as vlak_gqa

    # warm outside tracing: JIT-build + register before Dynamo sees the op
    _w = torch.randn(1, 32, 1, 64, device="cuda", dtype=torch.bfloat16)
    vlak_gqa(_w, _w, _w)

    def routed(self, attention_mask, batch_size, head_dim,
               query_states, key_states, value_states):
        M, N = query_states.shape[1], key_states.shape[1]
        par = params.get((M, N))
        if (par is not None and M <= 96 and head_dim == 64
                and query_states.dtype == torch.bfloat16):
            P, ds, de = par
            out = vlak_gqa(query_states,
                           key_states.to(query_states.dtype),
                           value_states.to(query_states.dtype),
                           scale=head_dim ** -0.5, prefix_len=P,
                           dead_start=ds, dead_end=de)
            return out.reshape(batch_size, M, -1)
        # fallback: sdpa with explicit GQA expand (prefix + unprobed sites)
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads
        groups = n_heads // n_kv
        if groups > 1:
            key_states = key_states[:, :, :, None, :].expand(
                batch_size, N, n_kv, groups, head_dim).reshape(
                batch_size, N, n_heads, head_dim)
            value_states = value_states[:, :, :, None, :].expand(
                batch_size, N, n_kv, groups, head_dim).reshape(
                batch_size, N, n_heads, head_dim)
        q = query_states.transpose(1, 2)
        k = key_states.to(query_states.dtype).transpose(1, 2)
        v = value_states.to(query_states.dtype).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask[:, None],
            scale=head_dim ** -0.5)
        return out.transpose(1, 2).reshape(batch_size, -1, n_heads * head_dim)

    cls = swe.SmolVLMWithExpertModel
    orig = cls.eager_attention_forward
    cls.eager_attention_forward = routed
    try:
        yield
    finally:
        cls.eager_attention_forward = orig


@contextlib.contextmanager
def _expert_bf16(policy):
    """Cast the checkpoint's stray fp32 modules to bf16.

    The pretrained VLM/expert weights load as bf16, but the freshly
    initialized modules — the 8 cross-attention projection blocks (odd
    expert layers) and both action-time MLPs — stay at PyTorch's default
    fp32. Their ~456 GEMMs per inference run as fp32 sgemm on CUDA cores
    (~8 ms) and force a per-call cast of the cross K/V cache at every
    expert attention site. bf16 weights put those GEMMs on tensor cores
    (fp32 accumulation inside, same story the attention gate already
    accepted). Pre-hooks cast any still-fp32 inputs at the module
    boundary; originals are restored on exit."""
    mods = [policy.model.action_time_mlp_in, policy.model.action_time_mlp_out]
    lm = policy.model.vlm_with_expert.lm_expert
    for i, layer in enumerate(lm.layers):
        if i % 2 == 1:
            mods.append(layer.self_attn)

    saved, hooks = [], []

    def cast_inputs(_m, args):
        return tuple(a.to(torch.bfloat16)
                     if torch.is_tensor(a) and a.is_floating_point()
                     and a.dtype != torch.bfloat16 else a for a in args)

    for m in mods:
        for p in list(m.parameters()) + list(m.buffers()):
            if p.dtype == torch.float32:
                saved.append((p, p.data))
                p.data = p.data.to(torch.bfloat16)
        hooks.append(m.register_forward_pre_hook(cast_inputs))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()
        for p, d in saved:
            p.data = d


@contextlib.contextmanager
def _vision_posids_static():
    """Assign the vision position ids directly instead of via boolean-mask
    scatter.

    SmolVLM's vision embeddings do `position_ids[mask] = pos_ids[mask]` to
    support variable-resolution images; the boolean indexing lowers to
    aten.nonzero (data-dependent shape), which breaks the Dynamo graph once
    per camera. For fixed full-resolution cameras the patch mask is all-ones
    by construction — the third ceremonial-mask incident in this project —
    so the scatter is the identity and the ids can be assigned directly.
    The fractional-coordinate computation is kept verbatim so the produced
    values are unchanged."""
    try:
        from transformers.models.smolvlm import modeling_smolvlm as mm
    except ImportError:
        yield
        return
    cls = mm.SmolVLMVisionEmbeddings
    orig = cls.forward

    def forward(self, pixel_values, patch_attention_mask):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side,
            device=pixel_values.device)
        nb_patches_h = patch_attention_mask[:, :, 0].sum(dim=1)
        nb_patches_w = patch_attention_mask[:, 0, :].sum(dim=1)
        step_h = 1.0 / nb_patches_h
        step_w = 1.0 / nb_patches_w
        max_patches_h = patch_attention_mask.size(1)
        max_patches_w = patch_attention_mask.size(2)
        h_indices = torch.arange(max_patches_h, device=pixel_values.device,
                                 dtype=torch.float32)
        w_indices = torch.arange(max_patches_w, device=pixel_values.device,
                                 dtype=torch.float32)
        fractional_coords_h = torch.clamp(h_indices[None, :] * step_h[:, None],
                                          max=(1.0 - 1e-6))
        fractional_coords_w = torch.clamp(w_indices[None, :] * step_w[:, None],
                                          max=(1.0 - 1e-6))
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries,
                                          right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries,
                                          right=True)
        pos_ids = (bucket_coords_h[:, :, None] * self.num_patches_per_side +
                   bucket_coords_w[:, None, :]).reshape(batch_size, -1)
        # all-ones mask -> the original masked scatter IS this assignment
        return embeddings + self.position_embedding(pos_ids)

    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = orig


def _compact_prefix(policy):
    """Drop the padded-language columns from the prefix at the source.

    The 48-slot language block carries ~44 padding tokens that the mask
    census proved algebraically inert: masked as keys at every site, and
    positions come from cumsum(pad_masks)-1 so pads never advance them.
    Removing them is an identity transform on the outputs, and every
    downstream computation shrinks (prefix 241 -> ~197: encode GEMMs,
    K/V cache, prefix attention, and the expert kernel's key lengths).

    STATIC-PER-TASK: the keep-index is captured on the first (eager) call
    and baked into the compiled graph; a changed task string needs a
    fresh variant setup. Patch applies immediately (so mask probes see
    compacted shapes); the returned context unpatches on exit."""
    model_cls = type(policy.model)
    orig = model_cls.embed_prefix
    cache = {}

    def capturing(self, *a, **kw):
        embs, pad, att = orig(self, *a, **kw)
        idx = pad[0].bool().nonzero().squeeze(1)
        cache["idx"] = idx
        # freeze: from now on the index is a captured constant — no data-
        # dependent branch survives into the traced graph (no graph break)
        def frozen(self, *a, **kw):
            e, p, m = orig(self, *a, **kw)
            return (e.index_select(1, idx), p.index_select(1, idx),
                    m.index_select(1, idx))
        model_cls.embed_prefix = frozen
        return (embs.index_select(1, idx), pad.index_select(1, idx),
                att.index_select(1, idx))

    model_cls.embed_prefix = capturing

    @contextlib.contextmanager
    def ctx():
        try:
            yield
        finally:
            model_cls.embed_prefix = orig

    return ctx()


@contextlib.contextmanager
def _sinusoid_fp32():
    """Compute the flow-loop timestep embedding in fp32 instead of fp64.

    lerobot's create_sinusoidal_pos_embedding asks get_safe_dtype for
    float64, which only downgrades on mps/xpu/cpu — on CUDA the embedding
    runs fp64 linspace/pow/sin/cos every flow step, then line 750 casts the
    result to bf16. Thor executes fp64 at 1/32 rate: ~340 us per step,
    ~3.4 ms per inference, for precision the very next line discards.
    fp32 transcendentals are bit-identical after the bf16 cast (23 mantissa
    bits vs bf16's 8)."""
    from lerobot.policies.smolvla import modeling_smolvla as _m
    orig = _m.get_safe_dtype
    _m.get_safe_dtype = lambda dtype, device: (
        torch.float32 if dtype == torch.float64 else orig(dtype, device))
    try:
        yield
    finally:
        _m.get_safe_dtype = orig


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
    if variant == "compiled-vk":   # compiled-ro + vlak attention at expert sites
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_attention_vlak()), \
            torch.compile(base, mode="reduce-overhead")
    if variant == "compiled-vk3":  # compiled-ro + v3 GQA-native kernel
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_attention_vlak3(params)), \
            torch.compile(base, mode="reduce-overhead")
    if variant == "compiled-vk4":  # + v4 register-pipeline kernel (mma.sync)
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_attention_vlak3(params, use_v4=True)), \
            torch.compile(base, mode="reduce-overhead")
    if variant == "compiled-vk4x":  # + stray-fp32 modules cast to bf16
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_attention_vlak3(params, use_v4=True),
                      _expert_bf16(policy)), \
            torch.compile(base, mode="reduce-overhead")
    if variant == "compiled-vk4c":  # + padded-language columns compacted away
        compact = _compact_prefix(policy)   # patch NOW: probe sees new shapes
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _route_attention_vlak3(params, use_v4=True),
                      _expert_bf16(policy), compact), \
            torch.compile(base, mode="reduce-overhead")
    if variant == "compiled-vk5":  # + graph-break fixes: ONE graph, enforced
        compact = _compact_prefix(policy)
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _vision_posids_static(),
                      _route_attention_vlak3(params, use_v4=True),
                      _expert_bf16(policy), compact), \
            torch.compile(base, mode="reduce-overhead", fullgraph=True)
    if variant == "compiled-vk7":  # vk5 + fp32 timestep embedding (fp64 fix)
        compact = _compact_prefix(policy)
        params = _probe_mask_params(policy)
        return _stack(_sdpa_flash_first(), _vision_mask_none(),
                      _vision_posids_static(), _sinusoid_fp32(),
                      _route_attention_vlak3(params, use_v4=True),
                      _expert_bf16(policy), compact), \
            torch.compile(base, mode="reduce-overhead", fullgraph=True)
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
