"""Parity tests for vlak::fused_attention_gqa (v3: GQA-native, strided
layouts, analytic two-boundary mask) against an fp32 reference."""
import pytest
import torch

from kernels.attention import fused_attention_gqa

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_attn(q, k, v, P, ds, de):
    """fp32 reference: expand GQA, build the analytic mask, math softmax."""
    B, M, H, D = q.shape
    N, Hkv = k.shape[1], k.shape[2]
    G = H // Hkv
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(G, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(G, dim=1)
    s = qf @ kf.transpose(-1, -2) * D ** -0.5
    r = torch.arange(M, device=q.device)[:, None]
    c = torch.arange(N, device=q.device)[None, :]
    vis = torch.where(c < P, (c < ds) | (c >= de), c <= (N - M) + r)
    s = s.masked_fill(~vis, float("-inf"))
    return (torch.softmax(s, -1) @ vf).permute(0, 2, 1, 3)


def check(q, k, v, P, ds, de):
    out = fused_attention_gqa(q, k, v, prefix_len=P,
                              dead_start=ds, dead_end=de).float()
    ref = ref_attn(q, k, v, P, ds, de)
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0).item()
    mx = (out - ref).abs().max().item()
    assert cos > 0.999 and mx < 3e-2, f"cos={cos} max_abs={mx}"
    assert out.shape == q.shape


def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


# (B, M, N, H, Hkv, P, ds, de) — the two real SmolVLA sites plus envelope
CASES = [
    (1, 50, 241, 15, 5, 241, 196, 240),   # expert cross, real mask
    (1, 50, 291, 15, 5, 241, 196, 240),   # expert self, real mask
    (1, 50, 241, 15, 5, 241, 0, 0),       # cross, no dead band
    (2, 37, 127, 8, 8, 100, 20, 44),      # MHA (G=1), odd sizes, tail tiles
    (1, 100, 256, 12, 4, 200, 64, 96),    # BMT=64 path, G=3
    (1, 32, 64, 4, 2, 64, 0, 0),          # exact tile boundaries
    (1, 33, 65, 4, 2, 40, 8, 16),         # one past tile boundaries
    (3, 50, 241, 15, 5, 241, 196, 240),   # batched
]


@cuda
@pytest.mark.parametrize("B,M,N,H,Hkv,P,ds,de", CASES)
def test_parity(B, M, N, H, Hkv, P, ds, de):
    torch.manual_seed(0)
    check(rand(B, M, H, 64), rand(B, N, Hkv, 64), rand(B, N, Hkv, 64),
          P, ds, de)


@cuda
def test_strided_inputs():
    """Slices of larger tensors: the layouts the model actually hands over."""
    torch.manual_seed(1)
    qbig = rand(1, 50, 17, 64)
    kbig = rand(1, 241, 7, 64)
    vbig = rand(1, 241, 7, 64)
    q, k, v = qbig[:, :, 1:16], kbig[:, :, 1:6], vbig[:, :, 2:7]
    assert not q.is_contiguous()
    check(q, k, v, 241, 196, 240)


@cuda
def test_matches_sdpa_unmasked():
    """No dead band + all-prefix == plain attention: cross-check vs F.sdpa."""
    torch.manual_seed(2)
    q, k, v = rand(1, 50, 15, 64), rand(1, 241, 5, 64), rand(1, 241, 5, 64)
    out = fused_attention_gqa(q, k, v, prefix_len=241).float()
    ke = k.repeat_interleave(3, dim=2)
    ve = v.repeat_interleave(3, dim=2)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), ke.transpose(1, 2), ve.transpose(1, 2)
    ).transpose(1, 2).float()
    assert torch.allclose(out, ref, rtol=2e-2, atol=2e-2)


@cuda
def test_composes_with_torch_compile():
    torch.manual_seed(3)
    q, k, v = rand(1, 50, 15, 64), rand(1, 241, 5, 64), rand(1, 241, 5, 64)
    eager = fused_attention_gqa(q, k, v, prefix_len=241,
                                dead_start=196, dead_end=240)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return fused_attention_gqa(q, k, v, prefix_len=241,
                                   dead_start=196, dead_end=240)

    out = fn(q, k, v)
    assert torch.equal(out, eager)


@cuda
def test_stress_randomized():
    """Regression for an intermittent wrong-result event observed once during
    development (cosine 0.07 after ~1500 calls, never reproduced; sanitizers
    and a 300-trial sweep were clean). Hammers the kernel with allocator
    churn and checks every trial against the fp32 reference."""
    fails = []
    for trial in range(30):
        torch.manual_seed(100 + trial)
        M, N = (50, 241) if trial % 2 == 0 else (50, 291)
        q = rand(1, 15, M, 64).transpose(1, 2)
        k, v = rand(1, N, 5, 64), rand(1, N, 5, 64)
        junk = [torch.empty(int(torch.randint(1000, 200000, (1,)).item()),
                            device="cuda", dtype=torch.bfloat16)
                for _ in range(4)]
        for _ in range(10):
            out = fused_attention_gqa(q, k, v, prefix_len=241,
                                      dead_start=196, dead_end=240)
        del junk
        ref = ref_attn(q, k, v, 241, 196, 240)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0).item()
        if cos < 0.999:
            fails.append((trial, cos))
    assert not fails, f"corrupted trials: {fails}"


@cuda
def test_rejects_bad_inputs():
    q, k, v = rand(1, 50, 15, 64), rand(1, 241, 5, 64), rand(1, 241, 5, 64)
    with pytest.raises(RuntimeError):
        fused_attention_gqa(q.float(), k.float(), v.float())  # not bf16
    with pytest.raises(RuntimeError):
        fused_attention_gqa(q, k, v, prefix_len=100,
                            dead_start=90, dead_end=120)      # de > P
    with pytest.raises(RuntimeError):
        fused_attention_gqa(q, rand(1, 241, 4, 64), rand(1, 241, 4, 64))
