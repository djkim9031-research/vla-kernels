"""Tests for vlak::fused_attention_gqa_v4s (swizzled dense smem).

v4s changes only the shared-memory addressing (Swizzle<3,3,3> over dense
64-element rows, replacing v4's 72-element padded pitch); atoms and
accumulation order are identical, so the oracle is bit-exact equality with
v4 — a stronger check than tolerance parity."""
import pytest
import torch

from kernels.attention import fused_attention_gqa_v4, fused_attention_gqa_v4s
from tests.test_attention_gqa import CASES, rand

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("B,M,N,H,Hkv,P,ds,de", CASES)
def test_bitexact_vs_v4(B, M, N, H, Hkv, P, ds, de):
    torch.manual_seed(0)
    q, k, v = rand(B, M, H, 64), rand(B, N, Hkv, 64), rand(B, N, Hkv, 64)
    a = fused_attention_gqa_v4(q, k, v, prefix_len=P,
                               dead_start=ds, dead_end=de)
    b = fused_attention_gqa_v4s(q, k, v, prefix_len=P,
                                dead_start=ds, dead_end=de)
    assert torch.equal(a, b)


@cuda
def test_bitexact_strided():
    torch.manual_seed(1)
    qb, kb, vb = rand(1, 50, 17, 64), rand(1, 241, 7, 64), rand(1, 241, 7, 64)
    q, k, v = qb[:, :, 1:16], kb[:, :, 1:6], vb[:, :, 2:7]
    assert not q.is_contiguous()
    a = fused_attention_gqa_v4(q, k, v, prefix_len=241,
                               dead_start=196, dead_end=240)
    b = fused_attention_gqa_v4s(q, k, v, prefix_len=241,
                                dead_start=196, dead_end=240)
    assert torch.equal(a, b)


@cuda
def test_unmasked_and_edge_shapes():
    torch.manual_seed(2)
    for (M, H, Hkv, N, P, ds, de) in [
        (32, 15, 5, 64, 64, 0, 0),       # exact tiles
        (33, 15, 5, 65, 65, 10, 20),     # every tail path
        (50, 15, 5, 241, -1, 0, 0),      # unmasked
    ]:
        q, k, v = rand(1, M, H, 64), rand(1, N, Hkv, 64), rand(1, N, Hkv, 64)
        pl = None if P < 0 else P
        a = fused_attention_gqa_v4(q, k, v, prefix_len=pl,
                                   dead_start=ds, dead_end=de)
        b = fused_attention_gqa_v4s(q, k, v, prefix_len=pl,
                                    dead_start=ds, dead_end=de)
        assert torch.equal(a, b)
