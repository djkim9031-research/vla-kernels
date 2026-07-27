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
VERSIONS = {"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4}
# "auto" picks empirically (see docs/softmax.md): warp-per-row (v1) wins for the
# many-row shapes typical of attention; v4 (fused online tree + raking combine
# + register-resident rows) covers the block-per-row regimes.
BEST = "auto"


def _auto_version(x: torch.Tensor) -> int:
    # measured crossovers (results/softmax.csv, locked clocks):
    #   wide rows  -> v4 wins regardless of row count (register-resident path)
    #   many rows, narrow/medium -> v1 (warp-per-row, no barriers)
    #   few rows   -> v4 (block-per-row is the only parallelism available)
    rows, cols = x.shape
    if cols > 2048:
        return VERSIONS["v4"]
    if rows >= 32:
        return VERSIONS["v1"]
    return VERSIONS["v4"]


@lru_cache(maxsize=1)
def _ext():
    ext = load(
        name="vlak_softmax",
        sources=[os.path.join(_THIS_DIR, "softmax_kernels.cu")],
        extra_cuda_cflags=["-O3", "-arch=native", "--use_fast_math"],
        verbose=bool(int(os.environ.get("VLAK_VERBOSE", "0"))),
    )
    # loading the .so ran TORCH_LIBRARY(vlak): register the fake (meta) impl
    # so FakeTensor tracing (torch.compile / torch.export) can shape-infer it
    # contiguous-stride promise, matching the real kernel's output
    torch.library.register_fake("vlak::softmax")(
        lambda x, version=4: torch.empty(x.shape, dtype=x.dtype, device=x.device))
    return ext


_registered = False


def _ensure_registered():
    # plain global-bool guard (not lru_cache) so Dynamo folds this branch away
    # instead of graph-breaking when tracing softmax() under torch.compile
    global _registered
    if not _registered:
        _ext()
        _registered = True


def softmax(x: torch.Tensor, version: str | int = BEST) -> torch.Tensor:
    """Softmax over the last dim of a 2D ``[rows, cols]`` CUDA tensor.

    Routed through the dispatcher-registered op (vlak::softmax), so calls
    compose with torch.compile / torch.export instead of graph-breaking.
    """
    if version == "auto":
        v = _auto_version(x)
    elif isinstance(version, str):
        v = VERSIONS[version]
    else:
        v = int(version)
    _ensure_registered()
    return torch.ops.vlak.softmax(x, v)


# ---- library baselines (cuDNN softmax, CUB-primitive softmax) --------------
# Built as a separate extension so the core kernels have no cuDNN dependency.
# cuDNN comes from the pip-bundled `nvidia-cudnn` shipped with torch; CUB from
# the CUDA toolkit's CCCL include tree.

@lru_cache(maxsize=1)
def _baselines_ext():
    import nvidia.cudnn

    cudnn_root = list(nvidia.cudnn.__path__)[0]  # namespace pkg: no __file__
    cccl_inc = "/usr/local/cuda/include/cccl"
    return load(
        name="vlak_softmax_baselines",
        sources=[os.path.join(_THIS_DIR, "softmax_baselines.cu")],
        extra_include_paths=[os.path.join(cudnn_root, "include"), cccl_inc],
        extra_cuda_cflags=["-O3", "-arch=native", "--use_fast_math"],
        extra_ldflags=[
            f"-L{os.path.join(cudnn_root, 'lib')}",
            f"-Wl,-rpath,{os.path.join(cudnn_root, 'lib')}",
            "-l:libcudnn.so.9",  # pip wheel has no unversioned .so symlink
        ],
        verbose=bool(int(os.environ.get("VLAK_VERBOSE", "0"))),
    )


def softmax_cudnn(x: torch.Tensor) -> torch.Tensor:
    """Vendor baseline: cudnnSoftmaxForward (ACCURATE, per-row)."""
    return _baselines_ext().softmax_cudnn(x)


def softmax_cub(x: torch.Tensor) -> torch.Tensor:
    """Library-primitive baseline: block-per-row softmax on cub::BlockReduce."""
    return _baselines_ext().softmax_cub(x)
