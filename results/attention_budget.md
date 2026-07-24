# Phase D step 1 — where compiled-ro's 73.8 ms actually goes

torch.profiler over the compiled-ro variant (3 inferences, locked clocks,
self-device-time accounting; kernel families sum to ~71 ms ≈ the 73.8 ms
wall, so the attribution is consistent). Eager run with all 44 attention
modules wrapped in profiler ranges provides the attention-vs-MLP GEMM split
(the extern cuBLAS calls are identical in both worlds).

## The budget

| family | ms/inf | share | notes |
|---|---|---|---|
| GEMM (cuBLAS/cutlass/nvjet) | 44.1 | ~60% | of which attention QKV/out-proj ≈ 8 ms (from eager attribution); the rest is MLP/backbone |
| vision SDPA (36 calls) | **11.4** | ~15% | `fmha_cutlassF` — see finding below |
| generated Triton (fused elementwise/norms) | 10.5 | ~14% | Inductor's own kernels |
| eager-attention fused softmax (Triton) | 2.1 | ~3% | the 176 fp32 sites, post-fusion |
| conv + misc | ~3 | ~4% | patch embed etc. |

Attention, all-in (SDPA + softmax + attention GEMMs): **≈ 21 ms of 73.8 ms (~29%)**.

## The surprise: compiled-ro picked a 3.2× slower SDPA backend

Same 36 vision-attention calls, same shapes (bf16, 12 heads × 1024 × 64):

- eager: `pytorch_flash::flash_fwd_kernel` — **3.55 ms/inf**
- compiled-ro: `fmha_cutlassF` (memory-efficient backend) — **11.24 ms/inf**

Inductor's lowering chose the mem-efficient kernel where eager's runtime
dispatcher chose flash. That is ~8 ms/inf left on the table by backend
selection alone — recoverable by forcing the flash backend
(`torch.nn.attention.sdpa_kernel`) or routing SDPA to our own kernel, before
any new CUDA is written.

## What this sets for Phase D

1. **Ceiling**: if all attention cost vanished, 73.8 → ~53 ms. Realistic
   target: ~60–66 ms (backend fix + fused-attention custom op on the eager
   fp32 sites + beating/matching flash on the vision sites).
2. **Order of attack**: (a) SDPA backend fix — measurement-only win, do
   first; (b) fused attention for the expert/prefix fp32 sites (softmax
   2.1 ms + attn-GEMM ~8 ms, replacing 3 launches + 2 HBM round-trips per
   site with one kernel); (c) only then consider competing with flash on the
   vision sites — flash at 3.55 ms is a hard target and (a) captures most of
   that value for free.
3. **Perspective for Phase E**: non-attention GEMMs (~36 ms) are the single
   biggest bucket — precision (bf16-native GEMM paths, INT8/FP8 via
   TensorRT) is the larger long-term lever than any attention work. Phase D
   is worth ~10–15 ms; quantization is worth more. Both are on the plan.
