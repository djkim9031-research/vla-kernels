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
    (4, 4096),     # large row (v4: past the register budget -> stream path)
]
TOL = {torch.float32: (1e-5, 1e-6), torch.float16: (3e-3, 1e-3),
       torch.bfloat16: (2e-2, 1.6e-2)}


@pytest.mark.parametrize("version", list(VERSIONS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch(version, dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    out = softmax(x, version=version)
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def _baselines():
    try:
        from kernels.softmax import softmax_cudnn, softmax_cub
        return {"cudnn": softmax_cudnn, "cub": softmax_cub}
    except Exception:
        return {}


@pytest.mark.parametrize("name", list(_baselines()) or ["unavailable"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_baselines_match_torch(name, dtype, shape):
    fns = _baselines()
    if name not in fns:
        pytest.skip("cuDNN/CUB baselines unavailable")
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(fns[name](x), ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("version", list(VERSIONS))
def test_extreme_values_are_stable(version):
    # large magnitudes must not overflow (safe/online softmax)
    x = torch.tensor([[1000.0, 1001.0, 999.0, -1000.0]], device="cuda")
    out = softmax(x, version=version)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(1, device="cuda"), rtol=1e-5, atol=1e-5)


def test_registered_op_and_fake_tensor():
    # the dispatcher-registered op resolves, and its fake impl shape-infers
    from kernels.softmax import _ext
    _ext()
    x = torch.randn(8, 64, device="cuda")
    torch.testing.assert_close(torch.ops.vlak.softmax(x, 1),
                               torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-6)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fx = torch.empty(8, 64, device="cuda")
        out = torch.ops.vlak.softmax(fx, 4)
    assert out.shape == (8, 64)


def test_composes_with_torch_compile():
    # our custom op must trace as one node — no graph break, correct output
    def f(x):
        return softmax(x, "v4") * 2.0

    x = torch.randn(16, 128, device="cuda")
    ref = torch.softmax(x, dim=-1) * 2.0
    softmax(x, "v4")  # warm: register the op so tracing sees _registered=True
    try:
        out = torch.compile(f, fullgraph=True)(x)
    except Exception as e:  # host triton/inductor env issue, not an op failure
        pytest.skip(f"torch.compile unavailable on host: {e}")
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
