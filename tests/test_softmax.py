"""Correctness: every softmax version must match torch.softmax over the last dim."""
import pytest
import torch

from kernels.softmax import softmax, VERSIONS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [
    (1, 16),       # tiny single row
    (8, 31),       # non-power-of-two cols
    (128, 64),
    (1024, 128),
    (256, 1000),   # cols > one block's threads (strided loop)
    (4, 4096),     # large row
]
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (3e-3, 1e-3)}


@pytest.mark.parametrize("version", list(VERSIONS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch(version, dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    out = softmax(x, version=version)
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("version", list(VERSIONS))
def test_extreme_values_are_stable(version):
    # large magnitudes must not overflow (safe/online softmax)
    x = torch.tensor([[1000.0, 1001.0, 999.0, -1000.0]], device="cuda")
    out = softmax(x, version=version)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(1, device="cuda"), rtol=1e-5, atol=1e-5)
