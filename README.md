# vla-kernels

Hand-written, Blackwell-tuned **CUDA/Triton kernels** for the hottest ops of a
**vision-language-action (VLA)** policy, with a reproducible profiling toolkit
that proves every speedup — and shows the policy stays accurate. Built and run on
**NVIDIA Jetson Thor**.

> Take a public VLA (**SmolVLA**), hand-write fused kernels for its hot ops, and
> prove — with roofline data and sim rollouts — that it runs materially faster on
> Thor at the same task accuracy.

## Why

GPU kernel craft (coalescing, warp/block reductions, online softmax, tensor-core
GEMM) shown on a *real* workload, with **speed _and_ accuracy** measured end to end
rather than microbenchmarks alone.

## Status

| component | state |
|-----------|-------|
| Kernel #1 — fused softmax (v0–v3 + auto) | ✅ built, tested, benchmarked → [docs/softmax.md](docs/softmax.md) |
| Library baselines — cuDNN softmax, CUB-reduction softmax | ✅ benchmarked + profiled alongside ours |
| Nsight Compute roofline analysis (all 6 kernels) | ✅ → [docs/softmax.md](docs/softmax.md) |
| Bench toolkit (correctness/latency/bandwidth, ncu parser, trtexec) | ✅ |
| SmolVLA harness (load / patch / parity / e2e) | ✅ code; runs in Docker |
| LayerNorm · fused attention · WMMA GEMM · Triton · INT8/TRT | ⏳ roadmap |

**Headline so far (locked clocks):** the warp-per-row kernel **matches cuDNN at
2048×512 and runs 1.8–2.0× faster at 512×2048** — up to **2.6× over
`torch.softmax`** (fp16) on the many-row shapes attention produces. Under
cold-cache profiling the CUB-reduction variant leads (22.7 µs vs cuDNN's
42.8 µs at 512×2048 fp32), and the counters explain every ranking — including
how cuDNN's register-resident kernel trades occupancy for minimal memory
traffic. Full analysis in [docs/softmax.md](docs/softmax.md).

## Layout

```
kernels/   hand-written CUDA (softmax done; layernorm/attention/gemm next) + common utils
triton/    Triton equivalents (CUDA-vs-Triton-vs-cuBLAS comparison) — roadmap
bench/     op-agnostic kernel bench, roofline classifier, trtexec parser, 3-tier driver, rig.sh
vla/       SmolVLA load + kernel patch + accuracy parity + end-to-end latency
tests/     pytest correctness (kernel == torch reference)
docs/      per-kernel optimization writeups
```

## Quickstart

Kernels build on the host toolchain (CUDA 13 / nvcc + ninja); the VLA/ML env
lives in Docker.

```bash
# kernels (host)
python3 -m pytest tests/test_softmax.py -q
python3 bench/bench_kernel.py --op softmax --dtypes fp32,fp16 --out results/softmax.csv

# determinism for clean numbers
bash bench/rig.sh

# VLA original-vs-tuned (Docker)
docker build -t vla-kernels .
docker run --rm -it --runtime nvidia -v "$PWD":/work vla-kernels \
    python3 vla/eval_accuracy.py --variant tuned --baseline results/acc_original.json
```

## Environment

Jetson Thor (sm_110, Blackwell) · CUDA 13.0 / nvcc · PyTorch 2.12+cu130 ·
Triton 3.7 · TensorRT (`trtexec`) · Nsight Systems/Compute. SmolVLA via
`lerobot` (`lerobot/smolvla_base`) in Docker.

## Roadmap

1. LayerNorm (Welford online) → 2. Fused flash-style attention (reuses online
softmax; biggest e2e win) → 3. Tiled → tensor-core (WMMA) GEMM vs cuBLAS →
4. Triton ports + comparison table → 5. TensorRT + INT8 path (stresses
accuracy parity) → 6. Cumulative end-to-end SmolVLA headline on Thor.
