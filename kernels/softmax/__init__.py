"""JIT-compiled fused row-softmax CUDA kernels.

The extension is built on first import via ``torch.utils.cpp_extension.load``
(no separate build step). Builds for the local GPU arch with ``-arch=native``
(resolves to sm_110 on Jetson Thor).
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# softmax version names -> int passed to the kernel dispatcher
VERSIONS = {"v0": 0, "v1": 1, "v2": 2, "v3": 3}
# "auto" picks empirically (see docs/softmax.md): warp-per-row (v1) wins for the
# many-row shapes typical of attention; block online (v3) wins for few wide rows.
BEST = "auto"


def _auto_version(x: torch.Tensor) -> int:
    rows, cols = x.shape
    if rows >= 32:          # enough rows to fill the GPU with warp-per-row
        return VERSIONS["v1"]
    return VERSIONS["v3"]   # few rows, wide: need intra-row parallelism + online


@lru_cache(maxsize=1)
def _ext():
    return load(
        name="vlak_softmax",
        sources=[os.path.join(_THIS_DIR, "softmax_kernels.cu")],
        extra_cuda_cflags=["-O3", "-arch=native", "--use_fast_math"],
        verbose=bool(int(os.environ.get("VLAK_VERBOSE", "0"))),
    )


def softmax(x: torch.Tensor, version: str | int = BEST) -> torch.Tensor:
    """Softmax over the last dim of a 2D ``[rows, cols]`` CUDA tensor."""
    if version == "auto":
        v = _auto_version(x)
    elif isinstance(version, str):
        v = VERSIONS[version]
    else:
        v = int(version)
    return _ext().softmax_lastdim(x, v)
