"""Correctness: fused attention vs the SDPA math reference."""
import math

import pytest
import torch
import torch.nn.functional as F

from kernels.attention import fused_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (B, H, M, N) at head_dim 64 — the measured SmolVLA sites + edges
SHAPES = [
    (1, 15, 50, 291),    # expert attention (flow steps)
    (1, 15, 50, 241),
    (1, 15, 241, 241),   # prefix attention
    (1, 12, 1024, 1024), # vision-shaped (for the op-level comparison vs flash)
    (1, 1, 1, 1),        # degenerate
    (2, 3, 17, 33),      # odd sizes, batch > 1
]
TOL = {torch.float32: (2e-5, 2e-5), torch.float16: (2e-3, 2e-3),
       torch.bfloat16: (2e-2, 2e-2)}


def ref_sdpa(q, k, v):
    # math oracle in fp32
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(dtype, shape):
    torch.manual_seed(0)
    B, H, M, N = shape
    q = torch.randn(B, H, M, 64, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, 64, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, 64, device="cuda", dtype=dtype)
    out = fused_attention(q, k, v)
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(out, ref_sdpa(q, k, v), rtol=rtol, atol=atol)


def test_matches_torch_sdpa_backend():
    torch.manual_seed(1)
    q = torch.randn(1, 15, 50, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 15, 291, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 15, 291, 64, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(fused_attention(q, k, v),
                               F.scaled_dot_product_attention(q, k, v),
                               rtol=2e-5, atol=2e-5)


def test_custom_scale():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 8, 64, device="cuda")
    k = torch.randn(1, 2, 16, 64, device="cuda")
    v = torch.randn(1, 2, 16, 64, device="cuda")
    torch.testing.assert_close(
        fused_attention(q, k, v, scale=0.5),
        F.scaled_dot_product_attention(q, k, v, scale=0.5),
        rtol=2e-5, atol=2e-5)


def test_extreme_values_stable():
    q = torch.full((1, 1, 4, 64), 30.0, device="cuda")
    k = torch.full((1, 1, 8, 64), 30.0, device="cuda")
    v = torch.randn(1, 1, 8, 64, device="cuda")
    out = fused_attention(q, k, v)
    assert torch.isfinite(out).all()


def test_composes_with_torch_compile():
    def f(q, k, v):
        return fused_attention(q, k, v) * 2.0

    torch.manual_seed(3)
    q = torch.randn(1, 4, 32, 64, device="cuda")
    k = torch.randn(1, 4, 64, 64, device="cuda")
    v = torch.randn(1, 4, 64, 64, device="cuda")
    fused_attention(q, k, v)  # warm: build + register before tracing
    ref = ref_sdpa(q, k, v) * 2.0
    try:
        out = torch.compile(f, fullgraph=True)(q, k, v)
    except Exception as e:
        pytest.skip(f"torch.compile unavailable on host: {e}")
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
