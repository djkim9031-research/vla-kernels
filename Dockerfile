# ML/VLA environment for vla-kernels (SmolVLA via lerobot + MLPerf LoadGen).
# Kernels build on the HOST toolchain; this image is for the lerobot/eval side.
#
# Thor / JetPack 7 (L4T r38) note: the l4t-jetpack container line stops at r36
# (JetPack 6). JetPack 7 uses the standard multi-arch CUDA images + aarch64
# (sbsa) PyTorch wheels — the same cu130 wheel line that runs on the host.
ARG BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv git ninja-build cmake g++ \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# venv to avoid PEP-668 friction on ubuntu24.04
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch cu130 aarch64 wheel (same line as the host's 2.12.0+cu130)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

# lerobot + SmolVLA extra
RUN pip install --no-cache-dir "lerobot[smolvla]" || \
    pip install --no-cache-dir lerobot

# MLCommons LoadGen for MLPerf-style SingleStream/Offline measurement.
# Needs Python.h (python3-dev) for the pybind11 bindings; installed here in a
# late layer so cache for the torch/lerobot layers above survives edits.
# Wheel first; source build as the aarch64 fallback.
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && (pip install --no-cache-dir mlcommons-loadgen || \
        (git clone --depth 1 https://github.com/mlcommons/inference.git /tmp/mlperf && \
         pip install --no-cache-dir /tmp/mlperf/loadgen && rm -rf /tmp/mlperf))

# The [smolvla] extra can fail transiently during the earlier layer's install
# (the || fallback then leaves plain lerobot). This layer guarantees the extra
# deps (transformers, accelerate, num2words) are present.
RUN pip install --no-cache-dir "lerobot[smolvla]"

# nvcc + CUDA headers so the tuned variant can JIT-build our kernels inside
# the container (the runtime base ships no compiler; NVIDIA's apt repo is
# preconfigured in the cuda images). Much smaller than switching to -devel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
# repo is bind-mounted at run time:  docker run -v "$PWD":/work ...
ENV PYTHONPATH=/work
CMD ["bash"]
