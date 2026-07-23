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
| SmolVLA harness (load / patch / parity / e2e / LoadGen) | ✅ measured on Thor |
| Variant comparison: eager · kernel-patch · torch.compile · +CUDA Graphs | ✅ → [results/e2e_comparison.md](results/e2e_comparison.md) |
| v4 softmax (bf16) · fused attention · WMMA GEMM · Triton · INT8/TRT | ⏳ roadmap |

**Headlines so far (locked clocks):**

- *Kernels:* v4 (fused online tree + raking combine + register-resident rows,
  fp32/fp16/**bf16**) leads every wide-row shape — **up to 3.5× over
  `torch.softmax` (bf16) and ahead of cuDNN and CUB**; v1 warp-per-row keeps
  the many-row/narrow crown. Registered as a dispatcher op
  (`vlak::softmax`), it traces as one node under
  `torch.compile(fullgraph=True)`. Nsight counters back every claim.
  → [docs/softmax.md](docs/softmax.md)
- *End-to-end (SmolVLA, 450M, bf16):* eager 198 ms/chunk → **73.8 ms with
  torch.compile + CUDA Graphs (2.7×, MLPerf-style p90 76.4 ms, accuracy
  retention 0.99997 — gate PASS)**. The per-op softmax swap alone is −3%:
  fusion granularity, not op substitution, is what survives end-to-end. The
  compiled number is the bar the hand-fused attention kernel must beat.
  → [results/e2e_comparison.md](results/e2e_comparison.md)

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
