"""Parity tests for vlak::fused_attention_gqa_v4 (register-pipeline mma.sync)
— same contract and case grid as the v3 op, plus a v3-equivalence check."""
import pytest
import torch

from kernels.attention import fused_attention_gqa, fused_attention_gqa_v4
from tests.test_attention_gqa import CASES, rand, ref_attn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("B,M,N,H,Hkv,P,ds,de", CASES)
def test_parity(B, M, N, H, Hkv, P, ds, de):
    torch.manual_seed(0)
    q, k, v = rand(B, M, H, 64), rand(B, N, Hkv, 64), rand(B, N, Hkv, 64)
    out = fused_attention_gqa_v4(q, k, v, prefix_len=P,
                                 dead_start=ds, dead_end=de).float()
    ref = ref_attn(q, k, v, P, ds, de)
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0).item()
    mx = (out - ref).abs().max().item()
    assert cos > 0.999 and mx < 3e-2, f"cos={cos} max_abs={mx}"


@cuda
def test_strided_inputs():
    torch.manual_seed(1)
    qb, kb, vb = rand(1, 50, 17, 64), rand(1, 241, 7, 64), rand(1, 241, 7, 64)
    q, k, v = qb[:, :, 1:16], kb[:, :, 1:6], vb[:, :, 2:7]
    assert not q.is_contiguous()
    out = fused_attention_gqa_v4(q, k, v, prefix_len=241,
                                 dead_start=196, dead_end=240).float()
    ref = ref_attn(q, k, v, 241, 196, 240)
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.999


@cuda
def test_matches_v3():
    """v4 must agree with v3 (same math, different engine) to bf16 noise."""
    torch.manual_seed(2)
    for N in (241, 291):
        q = rand(1, 50, 15, 64)
        k, v = rand(1, N, 5, 64), rand(1, N, 5, 64)
        a = fused_attention_gqa(q, k, v, prefix_len=241,
                                dead_start=196, dead_end=240).float()
        b = fused_attention_gqa_v4(q, k, v, prefix_len=241,
                                   dead_start=196, dead_end=240).float()
        assert torch.allclose(a, b, rtol=2e-2, atol=2e-2)


@cuda
def test_composes_with_torch_compile():
    torch.manual_seed(3)
    q, k, v = rand(1, 50, 15, 64), rand(1, 241, 5, 64), rand(1, 241, 5, 64)
    eager = fused_attention_gqa_v4(q, k, v, prefix_len=241,
                                   dead_start=196, dead_end=240)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return fused_attention_gqa_v4(q, k, v, prefix_len=241,
                                      dead_start=196, dead_end=240)

    assert torch.equal(fn(q, k, v), eager)


@cuda
def test_stress_randomized():
    fails = []
    for trial in range(30):
        torch.manual_seed(700 + trial)
        N = 241 if trial % 2 == 0 else 291
        q = rand(1, 15, 50, 64).transpose(1, 2)
        k, v = rand(1, N, 5, 64), rand(1, N, 5, 64)
        junk = [torch.empty(int(torch.randint(1000, 200000, (1,)).item()),
                            device="cuda", dtype=torch.bfloat16)
                for _ in range(4)]
        for _ in range(10):
            out = fused_attention_gqa_v4(q, k, v, prefix_len=241,
                                         dead_start=196, dead_end=240)
        del junk
        ref = ref_attn(q, k, v, 241, 196, 240)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0).item()
        if cos < 0.999:
            fails.append((trial, cos))
    assert not fails, f"corrupted trials: {fails}"
