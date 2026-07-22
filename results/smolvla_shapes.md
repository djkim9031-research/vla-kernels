# SmolVLA on Thor — measured interface, shapes, and baseline

First run of `lerobot/smolvla_base` (450.0M params) on Jetson Thor, in the
project Docker image (torch 2.11+cu130, lerobot 0.6.0). All numbers measured,
locked clocks.

## Model interface (per inference)

- Inputs: 3 cameras × (3,256,256), state (1,6), 48 language tokens
  (`observation.language.tokens` + bool attention mask — the lerobot 0.6
  pipeline pre-tokenizes; `vla/load_smolvla.py` replicates this)
- Output: action chunk (1, 50, 6) fp32; `select_action` pops one action per
  call from a queue and only runs the model every chunk_size-th call —
  benchmarks therefore time `predict_action_chunk` (the real recurring cost)
- Weights: **bfloat16**; VLM backbone SmolVLM2-500M-Video-Instruct, reduced
  to 16 layers
- The action expert is flow-matching: starting noise comes from the global
  RNG, so inference is stochastic (self-diff ~2.9 unseeded, exactly 0 when
  seeded) — parity tests seed before every call

## Attention geometry (hooked SDPA + F.softmax, one inference)

| site | path | shape | dtype | calls |
|---|---|---|---|---|
| SigLIP vision tower | SDPA | q,k = (1,12,1024,64) | bf16 | 36 (12 layers × 3 cams) |
| prefix/LLM attention | eager, F.softmax | (1,15,241,241) | **fp32** | 16 |
| action expert (flow steps) | eager, F.softmax | (1,15,50,291) + (1,15,50,241) | **fp32** | 80 + 80 |

241 prefix tokens = 192 image + 48 language + 1 state; 291 = prefix + 50
action queries. The flow-matching denoising loop re-runs expert attention
every step — 176 fp32 softmax calls per inference in total.

## Baseline (original, fp32-softmax patch off)

| metric | value |
|---|---|
| chunk latency p50 / p99 (CUDA events, 50 iters) | 215.6 / 221.5 ms |
| chunk rate | 4.6 Hz (≈232 amortized actions/s at chunk 50) |
| LoadGen SingleStream p90 (VALID) | 223.2 ms |
| LoadGen Offline | 4.63 samples/s (INVALID: 58.3 s < 60 s window; rerun with ≥290 queries) |
| parity reference (seeded, 16 obs) | max_abs_err 1.6e-2, cosine 0.999997 |

## Tuned row #1 — F.softmax swap only

219.0 ms p50 (~1.5% slower than original) at identical parity. The 176 eager
softmax tensors are small (≤0.9 MB), so per-call wrapper + launch overhead
outweighs the op-level kernel win. Conclusion, with data: swapping one op
inside eager attention is the wrong granularity — the fused attention kernel
(one launch replacing matmul → softmax → matmul) is the right lever, and the
vision tower additionally needs bf16 support to be addressable at all.

## Kernel-design consequences

1. Fused attention targets, in value order: (a) expert/prefix eager fp32
   attention — 176 calls, shapes above; (b) SigLIP SDPA — bf16, 36 calls.
2. All hand-written kernels need **bf16** variants to matter for this model.
3. Softmax rows here are the many-row regime — v1/warp-per-row territory,
   consistent with the op-level benchmark's regime table.
