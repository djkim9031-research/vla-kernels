# ML/VLA environment for vla-kernels (SmolVLA via lerobot).
# Kernels build on the HOST toolchain; this image is for the lerobot/sim side,
# per the "ML deps in Docker, repo bind-mounted" rule.
#
# Base must match the installed JetPack (R38 / JetPack 7, sm_110). NVIDIA's
# Jetson PyTorch container is the right base; set the tag to your JetPack:
#   nvcr.io/nvidia/l4t-jetpack:r38.x   (then pip install the matching torch)
# or a community dustynv/l4t-pytorch image carrying torch 2.x + cu13.
ARG BASE=nvcr.io/nvidia/l4t-jetpack:r38.4.0
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip git ninja-build && rm -rf /var/lib/apt/lists/*

# lerobot + SmolVLA extra. Pin a known-good lerobot release for API stability.
# (torch is expected from the base image; do not reinstall a non-Jetson wheel.)
RUN pip3 install --no-cache-dir "lerobot[smolvla]" || \
    pip3 install --no-cache-dir lerobot

WORKDIR /work
# repo is bind-mounted at run time:  docker run -v "$PWD":/work ...
ENV PYTHONPATH=/work
CMD ["bash"]
